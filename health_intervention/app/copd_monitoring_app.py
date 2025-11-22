"""
COPD Remote Patient Monitoring Application

Main application for patient daily symptom tracking, medication adherence monitoring,
and alert generation for clinical team.
"""

import sys
import os
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Tuple
import json
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.patient_models import (
    PatientDemographics,
    ClinicalProfile,
    DigitalLiteracyAssessment,
    CompletePatientProfile
)
from models.outcome_models import (
    CATAssessment,
    DailySymptomCheck,
    MedicationAdherence,
    AdherenceMetrics,
    HospitalizationEvent,
    StudyKPIs
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class COPDMonitoringApp:
    """
    COPD Remote Patient Monitoring Application

    Features:
    - Daily symptom check-ins (CAT score, breathlessness)
    - Medication adherence tracking
    - Exacerbation risk assessment
    - Alert generation for care team
    - Adherence metrics calculation
    """

    def __init__(self, patient_profile: CompletePatientProfile, data_dir: str = "./data"):
        """
        Initialize COPD monitoring app for a patient

        Args:
            patient_profile: Complete patient profile
            data_dir: Directory for storing patient data
        """
        self.patient = patient_profile
        self.patient_id = patient_profile.patient_id
        self.data_dir = Path(data_dir) / self.patient_id

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data stores
        self.symptom_checks: List[DailySymptomCheck] = []
        self.medication_adherence: List[MedicationAdherence] = []
        self.cat_assessments: List[CATAssessment] = []
        self.alerts: List[Dict] = []

        # Load existing data if available
        self._load_patient_data()

        logger.info(f"Initialized COPD Monitoring App for patient {self.patient_id}")

    def _load_patient_data(self):
        """Load existing patient data from disk"""
        symptom_file = self.data_dir / "symptom_checks.json"
        if symptom_file.exists():
            with open(symptom_file, 'r') as f:
                data = json.load(f)
                self.symptom_checks = [DailySymptomCheck(**item) for item in data]
                logger.info(f"Loaded {len(self.symptom_checks)} symptom checks")

    def _save_patient_data(self):
        """Save patient data to disk"""
        # Save symptom checks
        symptom_file = self.data_dir / "symptom_checks.json"
        with open(symptom_file, 'w') as f:
            data = [check.dict() for check in self.symptom_checks]
            json.dump(data, f, indent=2, default=str)

        # Save medication adherence
        med_file = self.data_dir / "medication_adherence.json"
        with open(med_file, 'w') as f:
            data = [med.dict() for med in self.medication_adherence]
            json.dump(data, f, indent=2, default=str)

        # Save CAT assessments
        cat_file = self.data_dir / "cat_assessments.json"
        with open(cat_file, 'w') as f:
            data = [cat.dict() for cat in self.cat_assessments]
            json.dump(data, f, indent=2, default=str)

        # Save alerts
        alert_file = self.data_dir / "alerts.json"
        with open(alert_file, 'w') as f:
            json.dump(self.alerts, f, indent=2, default=str)

        logger.info(f"Saved patient data to {self.data_dir}")

    def daily_symptom_checkin(
        self,
        breathlessness: int,
        cough: int,
        sputum: int,
        energy: int,
        worsening_symptoms: bool = False,
        increased_sputum_purulence: bool = False,
        fever: bool = False,
        chest_pain: bool = False,
        oxygen_saturation: Optional[int] = None,
        heart_rate: Optional[int] = None
    ) -> Dict:
        """
        Perform daily symptom check-in

        Args:
            breathlessness: Breathlessness severity (0-10)
            cough: Cough severity (0-10)
            sputum: Sputum production (0-10)
            energy: Energy level (0-10)
            worsening_symptoms: Whether symptoms worsening
            increased_sputum_purulence: Whether sputum more discolored
            fever: Whether patient has fever
            chest_pain: Whether patient has chest pain
            oxygen_saturation: SpO2 percentage
            heart_rate: Heart rate in bpm

        Returns:
            Dictionary with symptom check results and risk assessment
        """
        check = DailySymptomCheck(
            patient_id=self.patient_id,
            check_date=date.today(),
            timestamp=datetime.now(),
            breathlessness=breathlessness,
            cough=cough,
            sputum=sputum,
            energy=energy,
            worsening_symptoms=worsening_symptoms,
            increased_sputum_purulence=increased_sputum_purulence,
            fever=fever,
            chest_pain=chest_pain,
            oxygen_saturation=oxygen_saturation,
            heart_rate=heart_rate
        )

        self.symptom_checks.append(check)

        # Assess exacerbation risk
        risk_level = check.exacerbation_risk

        # Generate alert if high risk
        if "High risk" in risk_level:
            alert = self._generate_alert(
                alert_type="EXACERBATION_RISK",
                severity="HIGH",
                message=risk_level,
                data=check.dict()
            )
            logger.warning(f"HIGH RISK ALERT for {self.patient_id}: {risk_level}")
        elif "Moderate risk" in risk_level:
            alert = self._generate_alert(
                alert_type="EXACERBATION_RISK",
                severity="MODERATE",
                message=risk_level,
                data=check.dict()
            )
            logger.info(f"Moderate risk alert for {self.patient_id}: {risk_level}")
        else:
            alert = None

        # Save data
        self._save_patient_data()

        return {
            "success": True,
            "check_date": str(check.check_date),
            "exacerbation_risk": risk_level,
            "alert_generated": alert is not None,
            "alert": alert
        }

    def record_medication_adherence(
        self,
        controller_inhaler_taken: bool,
        rescue_inhaler_uses: int,
        correct_inhaler_technique: Optional[bool] = None,
        forgot: bool = False,
        side_effects: bool = False,
        cost_barrier: bool = False,
        ran_out: bool = False
    ) -> Dict:
        """
        Record medication adherence for today

        Args:
            controller_inhaler_taken: Whether long-acting inhaler taken
            rescue_inhaler_uses: Number of rescue inhaler uses
            correct_inhaler_technique: Whether technique correct
            forgot: Whether patient forgot to take medication
            side_effects: Whether experiencing side effects
            cost_barrier: Whether cost prevented taking medication
            ran_out: Whether patient ran out of medication

        Returns:
            Dictionary with adherence record confirmation
        """
        adherence = MedicationAdherence(
            patient_id=self.patient_id,
            date=date.today(),
            timestamp=datetime.now(),
            controller_inhaler_taken=controller_inhaler_taken,
            rescue_inhaler_uses=rescue_inhaler_uses,
            correct_inhaler_technique=correct_inhaler_technique,
            forgot=forgot,
            side_effects=side_effects,
            cost_barrier=cost_barrier,
            ran_out=ran_out
        )

        self.medication_adherence.append(adherence)

        # Check for concerning patterns
        alerts = []

        # Alert if medication not taken
        if not controller_inhaler_taken:
            reason = []
            if forgot:
                reason.append("forgot")
            if side_effects:
                reason.append("side effects")
            if cost_barrier:
                reason.append("cost barrier")
            if ran_out:
                reason.append("ran out")

            alert = self._generate_alert(
                alert_type="MEDICATION_NON_ADHERENCE",
                severity="MEDIUM",
                message=f"Controller medication not taken. Reasons: {', '.join(reason) if reason else 'none specified'}",
                data=adherence.dict()
            )
            alerts.append(alert)

        # Alert if excessive rescue inhaler use
        if rescue_inhaler_uses >= 4:
            alert = self._generate_alert(
                alert_type="EXCESSIVE_RESCUE_INHALER",
                severity="HIGH",
                message=f"Excessive rescue inhaler use: {rescue_inhaler_uses} times today",
                data=adherence.dict()
            )
            alerts.append(alert)
            logger.warning(f"Excessive rescue inhaler use for {self.patient_id}: {rescue_inhaler_uses} times")

        # Save data
        self._save_patient_data()

        return {
            "success": True,
            "date": str(adherence.date),
            "controller_taken": controller_inhaler_taken,
            "rescue_uses": rescue_inhaler_uses,
            "alerts_generated": len(alerts),
            "alerts": alerts
        }

    def complete_cat_assessment(
        self,
        cat_1_cough: int,
        cat_2_phlegm: int,
        cat_3_chest_tightness: int,
        cat_4_breathlessness: int,
        cat_5_activity_limitation: int,
        cat_6_confidence: int,
        cat_7_sleep: int,
        cat_8_energy: int
    ) -> Dict:
        """
        Complete full COPD Assessment Test (CAT)

        Args:
            cat_1_cough through cat_8_energy: CAT item scores (0-5 each)

        Returns:
            Dictionary with CAT results and change from baseline
        """
        assessment = CATAssessment(
            patient_id=self.patient_id,
            assessment_date=datetime.now(),
            source="App",
            cat_1_cough=cat_1_cough,
            cat_2_phlegm=cat_2_phlegm,
            cat_3_chest_tightness=cat_3_chest_tightness,
            cat_4_breathlessness=cat_4_breathlessness,
            cat_5_activity_limitation=cat_5_activity_limitation,
            cat_6_confidence=cat_6_confidence,
            cat_7_sleep=cat_7_sleep,
            cat_8_energy=cat_8_energy
        )

        self.cat_assessments.append(assessment)

        # Calculate change from baseline
        baseline_score = self.patient.clinical.baseline_cat_score
        change_metrics = None
        if baseline_score is not None:
            change_metrics = assessment.change_from_baseline(baseline_score)

            # Alert if significant worsening (MCID in wrong direction)
            if change_metrics['worsened']:
                alert = self._generate_alert(
                    alert_type="CAT_WORSENING",
                    severity="MEDIUM",
                    message=f"CAT score worsened by {change_metrics['change']} points (MCID achieved)",
                    data=assessment.dict()
                )
                logger.info(f"CAT score worsening for {self.patient_id}: {change_metrics['change']} points")

        # Save data
        self._save_patient_data()

        return {
            "success": True,
            "assessment_date": str(assessment.assessment_date),
            "total_score": assessment.total_score,
            "impact_category": assessment.impact_category,
            "baseline_score": baseline_score,
            "change_from_baseline": change_metrics
        }

    def calculate_adherence_metrics(
        self,
        start_date: date,
        end_date: date
    ) -> AdherenceMetrics:
        """
        Calculate adherence metrics for a date range

        Args:
            start_date: Start of period
            end_date: End of period

        Returns:
            AdherenceMetrics object
        """
        # Filter data to date range
        symptom_checks_in_range = [
            c for c in self.symptom_checks
            if start_date <= c.check_date <= end_date
        ]
        med_records_in_range = [
            m for m in self.medication_adherence
            if start_date <= m.date <= end_date
        ]

        # Count unique dates with each activity
        symptom_dates = set(c.check_date for c in symptom_checks_in_range)
        med_dates = set(m.date for m in med_records_in_range)

        # For demo purposes, assuming spirometry and activity sync
        # In production, these would come from actual device data
        total_days = (end_date - start_date).days + 1
        days_with_spirometry = int(total_days * 0.6)  # Placeholder
        days_with_activity_sync = int(total_days * 0.7)  # Placeholder

        # Calculate days with high engagement (≥3 of 4 activities)
        days_with_high_engagement = 0
        for day in range(total_days):
            current_date = start_date + timedelta(days=day)
            activities = sum([
                current_date in symptom_dates,
                current_date in med_dates,
                True if day % 2 == 0 else False,  # Placeholder for spirometry
                True if day % 3 != 0 else False   # Placeholder for activity
            ])
            if activities >= 3:
                days_with_high_engagement += 1

        metrics = AdherenceMetrics(
            patient_id=self.patient_id,
            start_date=start_date,
            end_date=end_date,
            total_days=total_days,
            days_with_symptom_check=len(symptom_dates),
            days_with_medication_confirmation=len(med_dates),
            days_with_spirometry=days_with_spirometry,
            days_with_activity_sync=days_with_activity_sync,
            days_with_high_engagement=days_with_high_engagement
        )

        return metrics

    def get_adherence_summary(self, days: int = 30) -> Dict:
        """
        Get adherence summary for last N days

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with adherence summary and KPI status
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        metrics = self.calculate_adherence_metrics(start_date, end_date)

        return {
            "patient_id": self.patient_id,
            "period": f"Last {days} days",
            "start_date": str(start_date),
            "end_date": str(end_date),
            "adherence_rate_percent": round(metrics.adherence_rate, 1),
            "adherence_category": metrics.adherence_category,
            "meets_kpi": metrics.meets_kpi_threshold,
            "engagement_by_activity": metrics.engagement_by_activity(),
            "days_with_high_engagement": metrics.days_with_high_engagement,
            "total_days": metrics.total_days
        }

    def _generate_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        data: Dict
    ) -> Dict:
        """
        Generate clinical alert

        Args:
            alert_type: Type of alert
            severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
            message: Alert message
            data: Supporting data

        Returns:
            Alert dictionary
        """
        alert = {
            "alert_id": f"ALERT-{len(self.alerts) + 1:04d}",
            "patient_id": self.patient_id,
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "acknowledged": False,
            "acknowledged_by": None,
            "acknowledged_at": None
        }

        self.alerts.append(alert)
        return alert

    def get_active_alerts(self) -> List[Dict]:
        """Get all unacknowledged alerts"""
        return [a for a in self.alerts if not a['acknowledged']]

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert

        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: Name/ID of person acknowledging

        Returns:
            True if successful
        """
        for alert in self.alerts:
            if alert['alert_id'] == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_by'] = acknowledged_by
                alert['acknowledged_at'] = datetime.now().isoformat()
                self._save_patient_data()
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        return False

    def get_symptom_trends(self, days: int = 30) -> Dict:
        """
        Get symptom trends over last N days

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with trend data
        """
        cutoff_date = date.today() - timedelta(days=days)
        recent_checks = [
            c for c in self.symptom_checks
            if c.check_date >= cutoff_date
        ]

        if not recent_checks:
            return {"error": "No symptom data available"}

        # Calculate averages
        avg_breathlessness = sum(c.breathlessness for c in recent_checks) / len(recent_checks)
        avg_cough = sum(c.cough for c in recent_checks) / len(recent_checks)
        avg_sputum = sum(c.sputum for c in recent_checks) / len(recent_checks)
        avg_energy = sum(c.energy for c in recent_checks) / len(recent_checks)

        # Calculate trend (comparing first half vs second half)
        mid_point = len(recent_checks) // 2
        first_half = recent_checks[:mid_point]
        second_half = recent_checks[mid_point:]

        trend = "stable"
        if first_half and second_half:
            avg_first = sum(c.breathlessness for c in first_half) / len(first_half)
            avg_second = sum(c.breathlessness for c in second_half) / len(second_half)

            if avg_second > avg_first + 1:
                trend = "worsening"
            elif avg_second < avg_first - 1:
                trend = "improving"

        return {
            "patient_id": self.patient_id,
            "period": f"Last {days} days",
            "total_checks": len(recent_checks),
            "averages": {
                "breathlessness": round(avg_breathlessness, 1),
                "cough": round(avg_cough, 1),
                "sputum": round(avg_sputum, 1),
                "energy": round(avg_energy, 1)
            },
            "trend": trend,
            "high_risk_days": sum(1 for c in recent_checks if "High risk" in c.exacerbation_risk)
        }

    def generate_patient_report(self, months: int = 6) -> Dict:
        """
        Generate comprehensive patient report for study KPIs

        Args:
            months: Number of months to include in report

        Returns:
            Dictionary with all KPI metrics
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=months * 30)

        # Adherence metrics
        adherence = self.calculate_adherence_metrics(start_date, end_date)

        # CAT score change
        baseline_cat = self.patient.clinical.baseline_cat_score
        recent_cats = [
            c for c in self.cat_assessments
            if c.assessment_date.date() >= start_date
        ]
        final_cat = recent_cats[-1].total_score if recent_cats else None

        cat_change = None
        achieved_mcid = False
        if baseline_cat is not None and final_cat is not None:
            cat_change = final_cat - baseline_cat
            achieved_mcid = cat_change <= -2

        # Symptom trends
        symptom_trends = self.get_symptom_trends(days=months * 30)

        # Active alerts
        active_alerts = self.get_active_alerts()

        return {
            "patient_id": self.patient_id,
            "report_period": f"{months} months",
            "start_date": str(start_date),
            "end_date": str(end_date),

            # Primary KPIs
            "kpis": {
                "adherence_rate_percent": round(adherence.adherence_rate, 1),
                "meets_adherence_kpi": adherence.meets_kpi_threshold,
                "baseline_cat_score": baseline_cat,
                "final_cat_score": final_cat,
                "cat_score_change": cat_change,
                "achieved_mcid": achieved_mcid
            },

            # Engagement metrics
            "engagement": {
                "total_symptom_checks": len([c for c in self.symptom_checks if c.check_date >= start_date]),
                "total_medication_records": len([m for m in self.medication_adherence if m.date >= start_date]),
                "total_cat_assessments": len(recent_cats),
                "days_with_high_engagement": adherence.days_with_high_engagement,
                "adherence_category": adherence.adherence_category
            },

            # Clinical status
            "clinical": {
                "symptom_trends": symptom_trends,
                "active_alerts": len(active_alerts),
                "high_priority_alerts": sum(1 for a in active_alerts if a['severity'] in ['HIGH', 'CRITICAL'])
            },

            # Equity profile
            "equity": {
                "race_ethnicity": self.patient.equity.race_ethnicity.value,
                "rurality": self.patient.equity.rurality.value,
                "digital_literacy": self.patient.equity.digital_literacy.value,
                "intersectionality_score": self.patient.equity.intersectionality_score,
                "equity_risk": self.patient.equity.risk_category
            }
        }


def demo_app_usage():
    """Demonstrate COPD monitoring app usage"""
    from models.patient_models import (
        PatientDemographics,
        ClinicalProfile,
        DigitalLiteracyAssessment,
        PatientEnrollment,
        EquityStratificationProfile,
        CompletePatientProfile,
        RaceEthnicity,
        RuralityCategory,
        COPDSeverity,
        DigitalLiteracyLevel
    )

    # Create sample patient profile
    demographics = PatientDemographics(
        patient_id="PT-DEMO-001",
        date_of_birth=date(1955, 3, 15),
        sex="Female",
        race_ethnicity=RaceEthnicity.BLACK,
        zip_code="43210",
        rurality=RuralityCategory.RURAL,
        insurance_type="Medicare"
    )

    clinical = ClinicalProfile(
        patient_id="PT-DEMO-001",
        copd_diagnosis_date=date(2018, 6, 10),
        copd_severity=COPDSeverity.GOLD_2,
        fev1_percent=65.2,
        fvc_percent=72.1,
        charlson_comorbidity_index=4,
        comorbidities=["I10", "E11.9"],
        baseline_cat_score=24,
        exacerbations_past_year=2,
        hospitalizations_past_year=1
    )

    digital_literacy = DigitalLiteracyAssessment(
        patient_id="PT-DEMO-001",
        assessment_date=date.today(),
        eheals_1=2, eheals_2=2, eheals_3=2, eheals_4=3,
        eheals_5=2, eheals_6=2, eheals_7=2, eheals_8=2
    )

    enrollment = PatientEnrollment(
        patient_id="PT-DEMO-001",
        enrollment_date=date.today() - timedelta(days=60),
        cluster_id="CLINIC-05",
        study_arm="Intervention",
        wedge_step=2
    )

    equity = EquityStratificationProfile(
        patient_id="PT-DEMO-001",
        race_ethnicity=RaceEthnicity.BLACK,
        rurality=RuralityCategory.RURAL,
        digital_literacy=DigitalLiteracyLevel.LOW,
        has_reliable_internet=False,
        has_smartphone=True,
        medicaid_eligible=False,
        lives_in_hpsa=True
    )

    patient_profile = CompletePatientProfile(
        demographics=demographics,
        clinical=clinical,
        digital_literacy=digital_literacy,
        enrollment=enrollment,
        equity=equity
    )

    # Initialize app
    app = COPDMonitoringApp(patient_profile, data_dir="./data")

    print("\n" + "="*60)
    print("COPD Remote Patient Monitoring App - Demo")
    print("="*60)

    # Daily symptom check
    print("\n1. Daily Symptom Check-in:")
    result = app.daily_symptom_checkin(
        breathlessness=6,
        cough=5,
        sputum=4,
        energy=4,
        worsening_symptoms=True,
        oxygen_saturation=92
    )
    print(json.dumps(result, indent=2, default=str))

    # Medication adherence
    print("\n2. Medication Adherence Recording:")
    result = app.record_medication_adherence(
        controller_inhaler_taken=True,
        rescue_inhaler_uses=2,
        correct_inhaler_technique=True
    )
    print(json.dumps(result, indent=2, default=str))

    # Complete CAT assessment
    print("\n3. CAT Assessment:")
    result = app.complete_cat_assessment(
        cat_1_cough=3, cat_2_phlegm=2, cat_3_chest_tightness=3,
        cat_4_breathlessness=4, cat_5_activity_limitation=3,
        cat_6_confidence=2, cat_7_sleep=3, cat_8_energy=2
    )
    print(json.dumps(result, indent=2, default=str))

    # Adherence summary
    print("\n4. Adherence Summary (Last 30 days):")
    summary = app.get_adherence_summary(days=30)
    print(json.dumps(summary, indent=2))

    # Generate patient report
    print("\n5. Comprehensive Patient Report:")
    report = app.generate_patient_report(months=2)
    print(json.dumps(report, indent=2))

    print("\n" + "="*60)
    print("Demo Complete")
    print("="*60)


if __name__ == "__main__":
    demo_app_usage()
