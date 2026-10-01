"""ZeroMalaria FastAPI — triage, referrals, alerts, analytics, sync."""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Annotated, Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth import assert_chw_own, get_current_user, require_roles, write_audit
from app.config import settings
from app.db import CaseRecord, Facility, FollowUp, Referral, SessionLocal, StockRecord, SyncEvent, User, get_db, init_db
from app.nlp import extract_symptoms_mock
from app.routers_ai import router as ai_router
from app.routers_auth import router as auth_router
from app.routers_auth import users_router
from app.routers_live import router as live_router
from app.routers_rbac import router as rbac_router
from app.routers_crud import router as crud_router
from app.roles import RBC_ADMIN, SUPER_ADMIN
from app.services.live_events import append_referral_event
from app.schemas import (
    ExtractRequest,
    HealthOut,
    ReferralCreate,
    ReferralOut,
    StatusUpdate,
    SyncRequest,
    SyncResponse,
    TriageRequest,
    TriageResponse,
)
from engine.decision import combine_decision
from engine.rules import decision_rank

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(rbac_router)
app.include_router(crud_router)
app.include_router(ai_router)
app.include_router(live_router)

_ANALYTICS_ROLES = (RBC_ADMIN, SUPER_ADMIN)

bearer_optional = HTTPBearer(auto_error=False)


def _optional_user(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_optional)],
    db: Session = Depends(get_db),
) -> User | None:
    if creds is None or not creds.credentials:
        return None
    from app.auth import decode_token

    data = decode_token(creds.credentials)
    return db.query(User).filter(User.id == data.get("sub")).first()


_PASSWORD_ENFORCE_ALLOW = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/auth/login",
    "/auth/demo-login",
    "/auth/refresh",
    "/auth/logout",
    "/auth/me",
    "/auth/change-password",
    "/auth/password-prompt/dismiss",
    "/auth/permissions",
}


@app.middleware("http")
async def password_enforce_middleware(request, call_next):
    """When ZM_PASSWORD_CHANGE_POLICY=enforce, limit API until password is changed."""
    from app.routers_auth import password_change_policy

    if password_change_policy() != "enforce":
        return await call_next(request)
    path = request.url.path
    if path in _PASSWORD_ENFORCE_ALLOW or path.startswith("/events"):
        return await call_next(request)
    auth = request.headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        return await call_next(request)
    token = auth.split(" ", 1)[1].strip()
    try:
        from app.auth import decode_token

        data = decode_token(token)
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == data.get("sub")).first()
            status = (getattr(user, "password_prompt_status", None) or "").lower() if user else ""
            if user and status == "pending":
                from fastapi.responses import JSONResponse

                return JSONResponse(status_code=403, content={"detail": "password_change_required"})
        finally:
            db.close()
    except Exception:
        pass
    return await call_next(request)


@app.on_event("startup")
def on_startup() -> None:
    # Refuse insecure defaults when not in demo mode.
    if not settings.demo_mode:
        defaults = {
            "zeromalaria-demo-secret-change-in-production",
            "change-me",
            "secret",
        }
        if settings.jwt_secret in defaults or settings.jwt_secret.startswith("zeromalaria-demo"):
            raise RuntimeError(
                "Refusing to start: ZM_DEMO_MODE=false but ZM_JWT_SECRET is still a demo default. "
                "Set a strong secret before production."
            )
        if settings.demo_password in {"demo1234", "password", "changeme"}:
            raise RuntimeError(
                "Refusing to start: ZM_DEMO_MODE=false but ZM_DEMO_PASSWORD is still a demo default."
            )
    init_db()
    if settings.auto_seed:
        db = SessionLocal()
        try:
            if db.query(User).count() == 0:
                from app.seed import run_seed

                run_seed(db, "full")
                db.commit()
        finally:
            db.close()


def _case_dict(body: TriageRequest) -> dict[str, Any]:
    return {
        "age_months": body.age_months,
        "sex": body.sex,
        "temperature_c": body.temperature_c,
        "fever_days": body.fever_days,
        "convulsions": int(body.convulsions),
        "unable_to_drink": int(body.unable_to_drink),
        "vomiting_everything": int(body.vomiting_everything),
        "lethargy": int(body.lethargy),
        "severe_breathing_difficulty": int(body.severe_breathing_difficulty),
        "tdr_result": body.tdr_result,
    }


def _referral_out(row: Referral, now: datetime | None = None) -> ReferralOut:
    now = now or datetime.utcnow()
    overdue = (
        row.status in {"sent", "received"}
        and row.arrived_at is None
        and (now - row.created_at) >= timedelta(hours=settings.overdue_hours)
    )
    return ReferralOut(
        id=row.id,
        client_uuid=row.client_uuid,
        facility_id=row.facility_id,
        chw_id=row.chw_id,
        district=row.district,
        sector=row.sector,
        age_months=row.age_months,
        sex=row.sex,
        decision=row.decision,
        reasons=json.loads(row.reasons_json or "[]"),
        summary=row.summary,
        status=row.status,
        created_at=row.created_at,
        received_at=row.received_at,
        arrived_at=row.arrived_at,
        treated_at=row.treated_at,
        demo_flag=row.demo_flag,
        overdue=overdue,
    )


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(
        status="ok",
        synthetic=True,
        disclaimer="Decision support tool. Not a replacement for clinical judgment.",
        demo_today=settings.demo_today,
    )


@app.get("/analytics/hotspots")
def analytics_hotspots(
    _user: Annotated[User, Depends(require_roles(*_ANALYTICS_ROLES))],
    district: Optional[str] = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Statistical signal only — never 'outbreak confirmed'."""
    q = db.query(CaseRecord)
    if district:
        q = q.filter(CaseRecord.district == district)
    rows = q.all()
    by_district: dict[str, list[CaseRecord]] = defaultdict(list)
    for r in rows:
        by_district[r.district].append(r)

    signals = []
    min_cases = 15
    for name, items in by_district.items():
        recent = [i for i in items if i.date >= "2026-09-01"]
        baseline = [i for i in items if "2026-06-01" <= i.date < "2026-09-01"]
        # Scale baseline to ~30-day equivalent (90 days -> /3)
        baseline_rate = len(baseline) / 3.0 if baseline else 0.0
        current = len(recent)
        if current < min_cases or baseline_rate <= 0:
            level = "none"
            pct = 0.0
        else:
            pct = round(((current - baseline_rate) / baseline_rate) * 100, 1)
            if pct >= 80:
                level = "high"
            elif pct >= 40:
                level = "moderate"
            elif pct >= 20:
                level = "low"
            else:
                level = "none"
        if level != "none":
            signals.append(
                {
                    "location": name,
                    "period": "2026-09 vs Jun–Aug baseline (synthetic)",
                    "current_cases": current,
                    "baseline_cases_monthly_equiv": round(baseline_rate, 1),
                    "percent_change": pct,
                    "signal_level": level,
                    "wording": "Potential increase detected (statistical signal)",
                    "recommended_action": "Verify with facility registers and CHW reports before any public claim.",
                    "evidence": {
                        "min_case_threshold": min_cases,
                        "method": "simple percent change vs prior 3-month monthly average",
                    },
                }
            )
    signals.sort(key=lambda s: s["percent_change"], reverse=True)
    return {
        "synthetic": True,
        "badge": settings.synthetic_badge,
        "signals": signals,
        "note": "Architecture demo on synthetic data — not outbreak confirmation.",
    }


@app.post("/triage", response_model=TriageResponse)
def triage(body: TriageRequest) -> TriageResponse:
    extracted = None
    case = _case_dict(body)
    if body.free_text:
        extracted = extract_symptoms_mock(body.free_text, body.language)
        suggested = extracted.get("suggested_fields") or {}
        for key, value in suggested.items():
            if key in case and isinstance(value, bool):
                case[key] = int(value)
            elif key == "temperature_c" and value is not None:
                case["temperature_c"] = float(value)
            elif key == "age_months" and value is not None:
                case["age_months"] = int(value)

    result = combine_decision(
        case,
        language=body.language,
        use_ml=body.use_ml,
        demo_scenario=body.demo_scenario,
    )
    payload = result.to_dict()
    payload["extracted_from_text"] = extracted
    payload["ml_threshold_treat_to_refer"] = 0.35
    return TriageResponse(**payload)


@app.post("/nlp/extract")
def nlp_extract(body: ExtractRequest) -> dict[str, Any]:
    # LLM flag: real API would go here; always fall back to mock for the hackathon.
    return extract_symptoms_mock(body.text, body.language)


def _create_referral(db: Session, body: ReferralCreate) -> tuple[Referral, bool]:
    existing = db.query(Referral).filter(Referral.client_uuid == body.client_uuid).first()
    if existing:
        return existing, True
    row = Referral(
        id=str(uuid.uuid4()),
        client_uuid=body.client_uuid,
        case_id=body.case_id,
        facility_id=body.facility_id,
        chw_id=body.chw_id,
        district=body.district,
        sector=body.sector,
        age_months=body.age_months,
        sex=body.sex,
        decision=body.decision,
        reasons_json=json.dumps(body.reasons),
        summary=body.summary,
        status="sent",
        created_at=datetime.utcnow(),
        demo_flag=False,
    )
    db.add(row)
    db.add(SyncEvent(client_uuid=body.client_uuid, payload_type="referral"))
    append_referral_event(db, "referral.created", row)
    db.commit()
    db.refresh(row)
    return row, False


@app.post("/referrals", response_model=ReferralOut)
def create_referral(body: ReferralCreate, db: Session = Depends(get_db)) -> ReferralOut:
    if decision_rank(body.decision) < decision_rank("refer"):
        raise HTTPException(400, "Only refer / urgent_refer can create a referral")
    row, _ = _create_referral(db, body)
    return _referral_out(row)


@app.get("/referrals", response_model=list[ReferralOut])
def list_referrals(
    facility_id: Optional[str] = None,
    chw_id: Optional[str] = None,
    db: Session = Depends(get_db),
) -> list[ReferralOut]:
    """Open list for demo sync; prefer /referrals/scoped with JWT in production."""
    q = db.query(Referral)
    if facility_id:
        q = q.filter(Referral.facility_id == facility_id)
    if chw_id:
        q = q.filter(Referral.chw_id == chw_id)
    rows = q.order_by(Referral.created_at.desc()).all()

    def sort_key(r: Referral):
        urgency = 0 if r.decision == "urgent_refer" else 1
        return (urgency, r.created_at)

    return [_referral_out(r) for r in sorted(rows, key=sort_key)]


@app.get("/referrals/scoped", response_model=list[ReferralOut])
def list_referrals_scoped(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ReferralOut]:
    q = db.query(Referral)
    from app.roles import CHW, HEALTH_CENTER, normalize_role

    role = normalize_role(user.role)
    if role == CHW:
        q = q.filter(Referral.chw_id == user.chw_code)
    elif role == HEALTH_CENTER:
        q = q.filter(Referral.facility_id == user.facility_id)
    # national roles: all
    rows = q.all()

    def sort_key(r: Referral):
        urgency = 0 if r.decision == "urgent_refer" else 1
        return (urgency, -(r.created_at.timestamp() if r.created_at else 0))

    return [_referral_out(r) for r in sorted(rows, key=sort_key)]


@app.patch("/referrals/{referral_id}/status", response_model=ReferralOut)
def patch_status(
    referral_id: str,
    body: StatusUpdate,
    db: Session = Depends(get_db),
    actor: User | None = Depends(_optional_user),
) -> ReferralOut:
    row = db.query(Referral).filter(Referral.id == referral_id).first()
    if not row:
        raise HTTPException(404, "Referral not found")
    now = datetime.utcnow()
    order = ["sent", "received", "arrived", "treated"]
    if order.index(body.status) < order.index(row.status if row.status in order else "sent"):
        raise HTTPException(400, "Cannot move status backwards")
    prev_status = row.status
    row.status = body.status
    if body.status == "received":
        row.received_at = row.received_at or now
    elif body.status == "arrived":
        row.received_at = row.received_at or now
        row.arrived_at = row.arrived_at or now
    elif body.status == "treated":
        row.received_at = row.received_at or now
        row.arrived_at = row.arrived_at or now
        row.treated_at = row.treated_at or now
        existing_fu = db.query(FollowUp).filter(FollowUp.referral_id == row.id).first()
        if not existing_fu:
            due = (now + timedelta(days=3)).date().isoformat()
            db.add(
                FollowUp(
                    id=str(uuid.uuid4()),
                    referral_id=row.id,
                    chw_id=row.chw_id,
                    facility_id=row.facility_id,
                    due_date=due,
                    status="due",
                    note="Post-treatment CHW follow-up (decision support reminder only).",
                    created_at=now,
                )
            )
    append_referral_event(
        db,
        "referral.status_changed",
        row,
        extra={"previous_status": prev_status, "new_status": body.status},
    )
    if actor:
        write_audit(
            db,
            action="referral_status",
            actor_id=actor.id,
            actor_username=actor.username,
            resource_type="referral",
            resource_id=row.id,
            detail=f"{prev_status}->{body.status}",
        )
    db.commit()
    db.refresh(row)
    return _referral_out(row)


@app.get("/alerts")
def alerts(chw_id: Optional[str] = None, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    now = datetime.utcnow()
    cutoff = now - timedelta(hours=settings.overdue_hours)
    q = db.query(Referral).filter(
        Referral.arrived_at.is_(None),
        Referral.status.in_(["sent", "received"]),
        Referral.created_at <= cutoff,
    )
    if chw_id:
        q = q.filter(Referral.chw_id == chw_id)
    rows = q.order_by(Referral.created_at.asc()).all()
    out = []
    for r in rows:
        ref = _referral_out(r, now).model_dump()
        out.append(
            {
                "id": ref.get("id") or r.id,
                "type": "referral_not_arrived",
                "message": "Patient has not arrived, follow up",
                "summary": ref.get("summary") or r.summary or "Patient has not arrived, follow up",
                "referral": ref,
            }
        )
    return out


def _compute_analytics_surge(
    db: Session,
    *,
    district: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    age_group: Optional[str] = None,
) -> dict[str, Any]:
    q = db.query(CaseRecord)
    if district:
        q = q.filter(CaseRecord.district == district)
    if date_from:
        q = q.filter(CaseRecord.date >= date_from)
    if date_to:
        q = q.filter(CaseRecord.date <= date_to)
    rows = q.all()
    if age_group == "under5":
        rows = [r for r in rows if r.age_months < 60]
    elif age_group == "5to14":
        rows = [r for r in rows if 60 <= r.age_months < 180]
    elif age_group == "15plus":
        rows = [r for r in rows if r.age_months >= 180]

    by_day: dict[str, int] = defaultdict(int)
    by_district: dict[str, int] = defaultdict(int)
    urgent = 0
    for r in rows:
        by_day[r.date] += 1
        by_district[r.district] += 1
        if r.decision == "urgent_refer":
            urgent += 1

    series = [{"date": d, "cases": by_day[d]} for d in sorted(by_day)]
    # Simple seasonal baseline forecast: average of same weekday over last 4 weeks
    forecast = []
    if series:
        last = datetime.fromisoformat(series[-1]["date"]).date()
        hist = {s["date"]: s["cases"] for s in series}
        for i in range(1, 15):
            day = last + timedelta(days=i)
            samples = []
            for w in range(1, 5):
                prev = (day - timedelta(days=7 * w)).isoformat()
                if prev in hist:
                    samples.append(hist[prev])
            mean = sum(samples) / len(samples) if samples else (series[-1]["cases"] if series else 0)
            forecast.append(
                {
                    "date": day.isoformat(),
                    "baseline": round(mean, 2),
                    "low": round(mean * 0.7, 2),
                    "high": round(mean * 1.3, 2),
                    "label": "baseline forecast",
                }
            )

    today = settings.demo_today
    cases_today = by_day.get(today, 0)
    return {
        "synthetic": True,
        "badge": settings.synthetic_badge,
        "cases_today": cases_today,
        "urgent_referrals": urgent,
        "total_cases": len(rows),
        "by_district": by_district,
        "series": series,
        "forecast": forecast,
        "note": "Synthetic demo data — not clinical performance",
    }


@app.get("/analytics/surge")
def analytics_surge(
    _user: Annotated[User, Depends(require_roles(*_ANALYTICS_ROLES))],
    district: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    age_group: Optional[str] = Query(None, description="under5|5to14|15plus"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return _compute_analytics_surge(
        db,
        district=district,
        date_from=date_from,
        date_to=date_to,
        age_group=age_group,
    )


@app.get("/analytics/stock")
def analytics_stock(
    _user: Annotated[User, Depends(require_roles(*_ANALYTICS_ROLES))],
    district: Optional[str] = None,
    week_start: Optional[str] = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    q = db.query(StockRecord)
    if district:
        q = q.filter(StockRecord.district == district)
    if week_start:
        q = q.filter(StockRecord.week_start == week_start)
    else:
        latest = db.query(func.max(StockRecord.week_start)).scalar()
        if latest:
            q = q.filter(StockRecord.week_start == latest)
            week_start = latest
    rows = q.all()
    cells = [
        {
            "facility_id": r.facility_id,
            "facility_name": r.facility_name,
            "district": r.district,
            "commodity": r.commodity,
            "stock_on_hand": r.stock_on_hand,
            "weeks_of_cover": r.weeks_of_cover,
            "stockout": bool(r.stockout),
            "risk": "stockout" if r.stockout else ("low" if r.weeks_of_cover < 2 else "ok"),
        }
        for r in rows
    ]
    return {
        "synthetic": True,
        "week_start": week_start,
        "cells": cells,
        "badge": settings.synthetic_badge,
    }


def _compute_analytics_funnel(db: Session, *, district: Optional[str] = None) -> dict[str, Any]:
    q = db.query(Referral)
    if district:
        q = q.filter(Referral.district == district)
    rows = q.all()
    referred = len(rows)
    received = sum(1 for r in rows if r.status in {"received", "arrived", "treated"} or r.received_at)
    arrived = sum(1 for r in rows if r.status in {"arrived", "treated"} or r.arrived_at)
    treated = sum(1 for r in rows if r.status == "treated" or r.treated_at)
    # Also blend case-level completion for broader funnel
    cq = db.query(CaseRecord).filter(CaseRecord.decision.in_(["refer", "urgent_refer"]))
    if district:
        cq = cq.filter(CaseRecord.district == district)
    case_refs = cq.count()
    case_arrived = cq.filter(CaseRecord.referral_completed == 1).count()
    delays = [
        r.arrival_delay_hours
        for r in cq.filter(CaseRecord.referral_completed == 1).all()
        if r.arrival_delay_hours is not None
    ]
    avg_delay = sum(delays) / len(delays) if delays else None
    completion_rate = (case_arrived / case_refs) if case_refs else None
    return {
        "synthetic": True,
        "live_referrals": {
            "referred": referred,
            "received": received,
            "arrived": arrived,
            "treated": treated,
            "lost_after_refer": max(0, referred - arrived),
        },
        "historical_cases": {
            "referred": case_refs,
            "arrived": case_arrived,
            "completion_rate": completion_rate,
            "avg_arrival_delay_hours": avg_delay,
        },
        "badge": settings.synthetic_badge,
    }


@app.get("/analytics/funnel")
def analytics_funnel(
    _user: Annotated[User, Depends(require_roles(*_ANALYTICS_ROLES))],
    district: Optional[str] = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return _compute_analytics_funnel(db, district=district)


@app.get("/analytics/kpis")
def analytics_kpis(
    _user: Annotated[User, Depends(require_roles(*_ANALYTICS_ROLES))],
    district: Optional[str] = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    surge = _compute_analytics_surge(db, district=district)
    funnel = _compute_analytics_funnel(db, district=district)
    alert_rows = alerts(db=db)
    return {
        "synthetic": True,
        "cases_today": surge["cases_today"],
        "urgent_referrals": surge["urgent_referrals"],
        "referral_completion_rate": funnel["historical_cases"]["completion_rate"],
        "avg_arrival_delay_hours": funnel["historical_cases"]["avg_arrival_delay_hours"],
        "active_alerts": len(alert_rows),
        "badge": settings.synthetic_badge,
        "disclaimer": "Decision support tool. Not a replacement for clinical judgment.",
    }


@app.get("/facilities")
def list_facilities(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.query(Facility).all()
    return [
        {
            "facility_id": r.facility_id,
            "name": r.name,
            "district": r.district,
            "sector": r.sector,
            "pilot": bool(r.pilot),
            "latitude": r.latitude,
            "longitude": r.longitude,
        }
        for r in rows
    ]


@app.post("/sync", response_model=SyncResponse)
def sync_batch(body: SyncRequest, db: Session = Depends(get_db)) -> SyncResponse:
    accepted = 0
    duplicates = 0
    results: list[dict[str, Any]] = []
    for item in body.items:
        if item.type == "referral":
            try:
                payload = ReferralCreate(**item.payload)
            except Exception as exc:
                results.append({"client_uuid": item.client_uuid, "ok": False, "error": str(exc)})
                continue
            if payload.client_uuid != item.client_uuid:
                payload.client_uuid = item.client_uuid
            row, dup = _create_referral(db, payload)
            if dup:
                duplicates += 1
            else:
                accepted += 1
            results.append({"client_uuid": item.client_uuid, "ok": True, "duplicate": dup, "id": row.id})
        else:
            # triage sync: store as idempotent event only for demo
            exists = db.query(SyncEvent).filter(SyncEvent.client_uuid == item.client_uuid).first()
            if exists:
                duplicates += 1
                results.append({"client_uuid": item.client_uuid, "ok": True, "duplicate": True})
            else:
                db.add(SyncEvent(client_uuid=item.client_uuid, payload_type="triage"))
                db.commit()
                accepted += 1
                results.append({"client_uuid": item.client_uuid, "ok": True, "duplicate": False})
    return SyncResponse(accepted=accepted, duplicates=duplicates, results=results)
