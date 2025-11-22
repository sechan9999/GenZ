"""
ETL Pipeline for COVID-19 Facility Data Analysis.

This module implements the Extract, Transform, Load pipeline that:
1. Extracts data from multiple sources (EHR, Staffing, PPE)
2. Performs fuzzy matching deduplication
3. Merges data sources on facility + date keys
4. Validates and cleans data
5. Loads to output data store

Handles ~1.2M rows daily from 170+ medical centers and 15,000 long-term care sites.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import hashlib

from models import (
    EHRRecord,
    StaffingRoster,
    PPEInventory,
    FacilityDailySnapshot,
    DataQualityIssue,
    DeduplicationMatch,
    FacilityType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FacilityDataETL:
    """
    ETL Pipeline for VA facility COVID-19 data.

    Processes daily data from multiple sources, performs deduplication,
    merges on facility + date keys, and generates merged snapshots.
    """

    def __init__(
        self,
        fuzzy_match_threshold: int = 85,
        date_tolerance_days: int = 0
    ):
        """
        Initialize ETL pipeline.

        Args:
            fuzzy_match_threshold: Minimum fuzzy match score (0-100) for facility name matching
            date_tolerance_days: Allow matching records within N days (0 = exact date match)
        """
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.date_tolerance_days = date_tolerance_days

        self.ehr_data: List[EHRRecord] = []
        self.staffing_data: List[StaffingRoster] = []
        self.ppe_data: List[PPEInventory] = []

        self.merged_snapshots: List[FacilityDailySnapshot] = []
        self.data_quality_issues: List[DataQualityIssue] = []
        self.deduplication_matches: List[DeduplicationMatch] = []

        # Facility name standardization cache
        self.facility_name_cache: Dict[str, str] = {}

        logger.info("ETL Pipeline initialized")

    # ========================================================================
    # EXTRACT: Load data from various sources
    # ========================================================================

    def extract_ehr_data(self, file_path: str) -> int:
        """
        Extract EHR data from CSV file.

        Args:
            file_path: Path to EHR CSV file

        Returns:
            Number of records extracted
        """
        logger.info(f"Extracting EHR data from {file_path}")

        try:
            df = pd.read_csv(file_path, parse_dates=['record_date'])

            for _, row in df.iterrows():
                try:
                    record = EHRRecord(
                        record_id=row['record_id'],
                        facility_id=row['facility_id'],
                        facility_name=row['facility_name'],
                        facility_type=row.get('facility_type', 'medical_center'),
                        record_date=pd.to_datetime(row['record_date']).date(),
                        total_patients=int(row['total_patients']),
                        covid_positive_count=int(row['covid_positive_count']),
                        covid_hospitalized=int(row['covid_hospitalized']),
                        covid_icu_count=int(row['covid_icu_count']),
                        covid_ventilator_count=int(row['covid_ventilator_count']),
                        covid_deaths=int(row.get('covid_deaths', 0)),
                        tests_conducted=int(row['tests_conducted']),
                        tests_positive=int(row['tests_positive']),
                        tests_pending=int(row.get('tests_pending', 0)),
                        first_dose_administered=int(row.get('first_dose_administered', 0)),
                        second_dose_administered=int(row.get('second_dose_administered', 0)),
                        booster_dose_administered=int(row.get('booster_dose_administered', 0)),
                    )
                    self.ehr_data.append(record)

                except Exception as e:
                    logger.warning(f"Skipping invalid EHR record {row.get('record_id', 'UNKNOWN')}: {e}")
                    self._log_data_quality_issue(
                        facility_id=row.get('facility_id', 'UNKNOWN'),
                        issue_type="validation_error",
                        severity="medium",
                        description=f"Invalid EHR record: {e}",
                        affected_fields=[]
                    )

            logger.info(f"Extracted {len(self.ehr_data)} EHR records")
            return len(self.ehr_data)

        except Exception as e:
            logger.error(f"Failed to extract EHR data: {e}")
            raise

    def extract_staffing_data(self, file_path: str) -> int:
        """
        Extract staffing roster data from CSV file.

        Args:
            file_path: Path to staffing CSV file

        Returns:
            Number of records extracted
        """
        logger.info(f"Extracting staffing data from {file_path}")

        try:
            df = pd.read_csv(file_path, parse_dates=['roster_date'])

            for _, row in df.iterrows():
                try:
                    roster = StaffingRoster(
                        roster_id=row['roster_id'],
                        facility_id=row['facility_id'],
                        facility_name=row['facility_name'],
                        roster_date=pd.to_datetime(row['roster_date']).date(),
                        physicians_scheduled=int(row['physicians_scheduled']),
                        physicians_present=int(row['physicians_present']),
                        nurses_scheduled=int(row['nurses_scheduled']),
                        nurses_present=int(row['nurses_present']),
                        respiratory_therapists_scheduled=int(row.get('respiratory_therapists_scheduled', 0)),
                        respiratory_therapists_present=int(row.get('respiratory_therapists_present', 0)),
                        support_staff_scheduled=int(row.get('support_staff_scheduled', 0)),
                        support_staff_present=int(row.get('support_staff_present', 0)),
                        staff_covid_positive=int(row.get('staff_covid_positive', 0)),
                        staff_quarantined=int(row.get('staff_quarantined', 0)),
                        staff_vaccinated_full=int(row.get('staff_vaccinated_full', 0)),
                        total_beds=int(row['total_beds']),
                        occupied_beds=int(row['occupied_beds']),
                        covid_beds_available=int(row.get('covid_beds_available', 0)),
                        icu_beds_total=int(row.get('icu_beds_total', 0)),
                        icu_beds_occupied=int(row.get('icu_beds_occupied', 0)),
                    )
                    self.staffing_data.append(roster)

                except Exception as e:
                    logger.warning(f"Skipping invalid staffing record {row.get('roster_id', 'UNKNOWN')}: {e}")
                    self._log_data_quality_issue(
                        facility_id=row.get('facility_id', 'UNKNOWN'),
                        issue_type="validation_error",
                        severity="medium",
                        description=f"Invalid staffing record: {e}",
                        affected_fields=[]
                    )

            logger.info(f"Extracted {len(self.staffing_data)} staffing records")
            return len(self.staffing_data)

        except Exception as e:
            logger.error(f"Failed to extract staffing data: {e}")
            raise

    def extract_ppe_data(self, file_path: str) -> int:
        """
        Extract PPE inventory data from CSV file.

        Args:
            file_path: Path to PPE CSV file

        Returns:
            Number of records extracted
        """
        logger.info(f"Extracting PPE inventory data from {file_path}")

        try:
            df = pd.read_csv(file_path, parse_dates=['inventory_date'])

            for _, row in df.iterrows():
                try:
                    inventory = PPEInventory(
                        inventory_id=row['inventory_id'],
                        facility_id=row['facility_id'],
                        facility_name=row['facility_name'],
                        inventory_date=pd.to_datetime(row['inventory_date']).date(),
                        n95_masks_count=int(row['n95_masks_count']),
                        surgical_masks_count=int(row['surgical_masks_count']),
                        face_shields_count=int(row.get('face_shields_count', 0)),
                        gowns_count=int(row['gowns_count']),
                        gloves_boxes=int(row['gloves_boxes']),
                        hand_sanitizer_bottles=int(row.get('hand_sanitizer_bottles', 0)),
                        disinfectant_wipes_count=int(row.get('disinfectant_wipes_count', 0)),
                        ventilators_total=int(row.get('ventilators_total', 0)),
                        ventilators_in_use=int(row.get('ventilators_in_use', 0)),
                        n95_days_supply=float(row.get('n95_days_supply', 0)) if pd.notna(row.get('n95_days_supply')) else None,
                        surgical_mask_days_supply=float(row.get('surgical_mask_days_supply', 0)) if pd.notna(row.get('surgical_mask_days_supply')) else None,
                        gown_days_supply=float(row.get('gown_days_supply', 0)) if pd.notna(row.get('gown_days_supply')) else None,
                        critical_shortage=bool(row.get('critical_shortage', False)),
                        reorder_needed=str(row.get('reorder_needed', '')).split(',') if row.get('reorder_needed') else [],
                    )
                    self.ppe_data.append(inventory)

                except Exception as e:
                    logger.warning(f"Skipping invalid PPE record {row.get('inventory_id', 'UNKNOWN')}: {e}")
                    self._log_data_quality_issue(
                        facility_id=row.get('facility_id', 'UNKNOWN'),
                        issue_type="validation_error",
                        severity="medium",
                        description=f"Invalid PPE record: {e}",
                        affected_fields=[]
                    )

            logger.info(f"Extracted {len(self.ppe_data)} PPE inventory records")
            return len(self.ppe_data)

        except Exception as e:
            logger.error(f"Failed to extract PPE data: {e}")
            raise

    # ========================================================================
    # TRANSFORM: Fuzzy matching, deduplication, standardization
    # ========================================================================

    def standardize_facility_name(self, facility_name: str) -> str:
        """
        Standardize facility names using fuzzy matching.

        Uses cached standardization to ensure consistency across records.

        Args:
            facility_name: Raw facility name

        Returns:
            Standardized facility name
        """
        # Check cache first
        if facility_name in self.facility_name_cache:
            return self.facility_name_cache[facility_name]

        # Clean the name
        cleaned = facility_name.strip().upper()

        # Common standardizations
        replacements = {
            'VAMC': 'VA MEDICAL CENTER',
            'VA MED CTR': 'VA MEDICAL CENTER',
            'VA MED CENTER': 'VA MEDICAL CENTER',
            'VETERAN AFFAIRS': 'VETERANS AFFAIRS',
            'HEALTHCARE SYS': 'HEALTHCARE SYSTEM',
            'HLTHCARE': 'HEALTHCARE',
            'CLC': 'COMMUNITY LIVING CENTER',
        }

        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        # Cache the result
        self.facility_name_cache[facility_name] = cleaned
        return cleaned

    def fuzzy_match_facilities(
        self,
        facility_name_1: str,
        facility_name_2: str
    ) -> Tuple[bool, int]:
        """
        Perform fuzzy matching on facility names.

        Uses token_sort_ratio to handle word order variations.

        Args:
            facility_name_1: First facility name
            facility_name_2: Second facility name

        Returns:
            Tuple of (is_match, match_score)
        """
        # Standardize both names first
        std_name_1 = self.standardize_facility_name(facility_name_1)
        std_name_2 = self.standardize_facility_name(facility_name_2)

        # Calculate fuzzy match score
        match_score = fuzz.token_sort_ratio(std_name_1, std_name_2)

        is_match = match_score >= self.fuzzy_match_threshold

        return is_match, match_score

    def deduplicate_records(self) -> Dict[str, int]:
        """
        Deduplicate records using fuzzy matching on facility + date keys.

        Identifies and removes duplicate records based on:
        - Fuzzy facility name matching
        - Exact or near-exact date matching

        Returns:
            Dictionary with deduplication statistics
        """
        logger.info("Starting deduplication process...")

        stats = {
            'ehr_duplicates': 0,
            'staffing_duplicates': 0,
            'ppe_duplicates': 0,
            'total_duplicates': 0
        }

        # Deduplicate EHR records
        stats['ehr_duplicates'] = self._deduplicate_list(
            self.ehr_data,
            id_field='record_id',
            facility_field='facility_name',
            date_field='record_date',
            data_type='EHR'
        )

        # Deduplicate Staffing records
        stats['staffing_duplicates'] = self._deduplicate_list(
            self.staffing_data,
            id_field='roster_id',
            facility_field='facility_name',
            date_field='roster_date',
            data_type='Staffing'
        )

        # Deduplicate PPE records
        stats['ppe_duplicates'] = self._deduplicate_list(
            self.ppe_data,
            id_field='inventory_id',
            facility_field='facility_name',
            date_field='inventory_date',
            data_type='PPE'
        )

        stats['total_duplicates'] = sum([
            stats['ehr_duplicates'],
            stats['staffing_duplicates'],
            stats['ppe_duplicates']
        ])

        logger.info(f"Deduplication complete. Removed {stats['total_duplicates']} duplicates")
        return stats

    def _deduplicate_list(
        self,
        records: List,
        id_field: str,
        facility_field: str,
        date_field: str,
        data_type: str
    ) -> int:
        """
        Deduplicate a list of records using fuzzy matching.

        Args:
            records: List of record objects
            id_field: Field name for unique ID
            facility_field: Field name for facility name
            date_field: Field name for date
            data_type: Type of data for logging

        Returns:
            Number of duplicates removed
        """
        if not records:
            return 0

        duplicates_found = []
        seen_keys = {}  # {standardized_key: record_index}

        for i, record in enumerate(records):
            facility_name = getattr(record, facility_field)
            record_date = getattr(record, date_field)

            # Generate composite key
            std_facility = self.standardize_facility_name(facility_name)
            composite_key = f"{std_facility}|{record_date}"

            # Check for duplicates
            if composite_key in seen_keys:
                # Found duplicate
                original_idx = seen_keys[composite_key]
                original_record = records[original_idx]

                # Log deduplication match
                match_id = hashlib.md5(composite_key.encode()).hexdigest()[:16]
                self.deduplication_matches.append(
                    DeduplicationMatch(
                        match_id=match_id,
                        record_1_id=getattr(original_record, id_field),
                        record_2_id=getattr(record, id_field),
                        match_score=1.0,  # Exact match
                        matched_on=[facility_field, date_field],
                        action_taken="removed_duplicate"
                    )
                )

                duplicates_found.append(i)
                logger.debug(f"Duplicate {data_type} record found: {getattr(record, id_field)}")

            else:
                # Check fuzzy matches for near-duplicates
                for existing_key, existing_idx in seen_keys.items():
                    existing_facility, existing_date_str = existing_key.split('|')
                    existing_date = datetime.strptime(existing_date_str, '%Y-%m-%d').date()

                    # Check if dates are within tolerance
                    date_diff = abs((record_date - existing_date).days)

                    if date_diff <= self.date_tolerance_days:
                        # Perform fuzzy matching on facility names
                        is_match, match_score = self.fuzzy_match_facilities(
                            facility_name,
                            records[existing_idx].facility_name
                        )

                        if is_match:
                            # Found fuzzy duplicate
                            match_id = hashlib.md5(f"{composite_key}_{existing_key}".encode()).hexdigest()[:16]
                            self.deduplication_matches.append(
                                DeduplicationMatch(
                                    match_id=match_id,
                                    record_1_id=getattr(records[existing_idx], id_field),
                                    record_2_id=getattr(record, id_field),
                                    match_score=match_score / 100.0,
                                    matched_on=[facility_field, date_field],
                                    action_taken="removed_fuzzy_duplicate"
                                )
                            )

                            duplicates_found.append(i)
                            logger.debug(f"Fuzzy duplicate {data_type} record found (score={match_score}): {getattr(record, id_field)}")
                            break

                # If not a duplicate, add to seen keys
                if i not in duplicates_found:
                    seen_keys[composite_key] = i

        # Remove duplicates (in reverse order to preserve indices)
        for idx in sorted(duplicates_found, reverse=True):
            del records[idx]

        logger.info(f"Removed {len(duplicates_found)} duplicate {data_type} records")
        return len(duplicates_found)

    # ========================================================================
    # MERGE: Combine data sources on facility + date keys
    # ========================================================================

    def merge_data_sources(self) -> int:
        """
        Merge EHR, Staffing, and PPE data on facility + date keys.

        Creates FacilityDailySnapshot objects with data from all three sources.

        Returns:
            Number of merged snapshots created
        """
        logger.info("Merging data sources...")

        # Create dictionaries for fast lookups
        ehr_dict = self._create_lookup_dict(self.ehr_data, 'facility_id', 'record_date')
        staffing_dict = self._create_lookup_dict(self.staffing_data, 'facility_id', 'roster_date')
        ppe_dict = self._create_lookup_dict(self.ppe_data, 'facility_id', 'inventory_date')

        # Get all unique (facility, date) combinations
        all_keys = set(ehr_dict.keys()) | set(staffing_dict.keys()) | set(ppe_dict.keys())

        for key in all_keys:
            facility_id, snapshot_date = key.split('|')
            snapshot_date = datetime.strptime(snapshot_date, '%Y-%m-%d').date()

            # Get data from each source
            ehr_record = ehr_dict.get(key)
            staffing_record = staffing_dict.get(key)
            ppe_record = ppe_dict.get(key)

            # Determine facility name (prefer EHR, then staffing, then PPE)
            facility_name = (
                ehr_record.facility_name if ehr_record else
                staffing_record.facility_name if staffing_record else
                ppe_record.facility_name if ppe_record else
                "UNKNOWN"
            )

            # Determine facility type
            facility_type = (
                ehr_record.facility_type if ehr_record else
                FacilityType.MEDICAL_CENTER
            )

            # Calculate merge confidence
            merge_confidence = self._calculate_merge_confidence(
                ehr_record, staffing_record, ppe_record
            )

            # Track data sources used
            data_sources_used = []
            if ehr_record:
                data_sources_used.append("VistA EHR")
            if staffing_record:
                data_sources_used.append("HRIS")
            if ppe_record:
                data_sources_used.append("MERS")

            # Create merged snapshot
            snapshot_id = f"SNAPSHOT-{snapshot_date.strftime('%Y%m%d')}-{facility_id}"

            snapshot = FacilityDailySnapshot(
                snapshot_id=snapshot_id,
                facility_id=facility_id,
                facility_name=self.standardize_facility_name(facility_name),
                facility_type=facility_type,
                snapshot_date=snapshot_date,
                ehr_data=ehr_record,
                staffing_data=staffing_record,
                ppe_data=ppe_record,
                merge_confidence=merge_confidence,
                data_sources_used=data_sources_used
            )

            # Calculate derived metrics
            snapshot.calculate_derived_metrics()

            self.merged_snapshots.append(snapshot)

        logger.info(f"Created {len(self.merged_snapshots)} merged snapshots")
        return len(self.merged_snapshots)

    def _create_lookup_dict(
        self,
        records: List,
        facility_field: str,
        date_field: str
    ) -> Dict[str, any]:
        """
        Create lookup dictionary for records keyed by facility_id|date.

        Args:
            records: List of record objects
            facility_field: Field name for facility ID
            date_field: Field name for date

        Returns:
            Dictionary mapping composite key to record
        """
        lookup = {}
        for record in records:
            facility_id = getattr(record, facility_field)
            record_date = getattr(record, date_field)
            key = f"{facility_id}|{record_date}"
            lookup[key] = record
        return lookup

    def _calculate_merge_confidence(
        self,
        ehr_record: Optional[EHRRecord],
        staffing_record: Optional[StaffingRoster],
        ppe_record: Optional[PPEInventory]
    ) -> float:
        """
        Calculate confidence score for merged record.

        Higher confidence when:
        - All three sources present
        - Facility names match closely
        - Data values are consistent

        Args:
            ehr_record: EHR record (optional)
            staffing_record: Staffing record (optional)
            ppe_record: PPE record (optional)

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.0

        # Base confidence from data source availability
        sources_present = sum([
            1 if ehr_record else 0,
            1 if staffing_record else 0,
            1 if ppe_record else 0
        ])
        confidence += (sources_present / 3.0) * 0.5  # Max 0.5 from availability

        # Bonus confidence from facility name consistency
        names = []
        if ehr_record:
            names.append(ehr_record.facility_name)
        if staffing_record:
            names.append(staffing_record.facility_name)
        if ppe_record:
            names.append(ppe_record.facility_name)

        if len(names) >= 2:
            # Check if all names match
            std_names = [self.standardize_facility_name(n) for n in names]
            if len(set(std_names)) == 1:
                confidence += 0.3  # All names match exactly
            else:
                # Calculate average fuzzy match score
                match_scores = []
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        _, score = self.fuzzy_match_facilities(names[i], names[j])
                        match_scores.append(score)
                if match_scores:
                    avg_score = sum(match_scores) / len(match_scores)
                    confidence += (avg_score / 100.0) * 0.3

        # Bonus confidence from data consistency (e.g., ventilator counts)
        if ehr_record and ppe_record:
            ehr_vents = ehr_record.covid_ventilator_count
            ppe_vents = ppe_record.ventilators_in_use
            if ehr_vents == ppe_vents:
                confidence += 0.2  # Ventilator counts match

        return min(confidence, 1.0)

    # ========================================================================
    # LOAD: Export merged data
    # ========================================================================

    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export merged snapshots to pandas DataFrame.

        Returns:
            DataFrame with flattened snapshot data
        """
        logger.info("Exporting merged snapshots to DataFrame...")

        records = []
        for snapshot in self.merged_snapshots:
            record = {
                'snapshot_id': snapshot.snapshot_id,
                'facility_id': snapshot.facility_id,
                'facility_name': snapshot.facility_name,
                'facility_type': snapshot.facility_type,
                'snapshot_date': snapshot.snapshot_date,
                'occupancy_rate': snapshot.occupancy_rate,
                'staff_shortage': snapshot.staff_shortage,
                'ppe_critical': snapshot.ppe_critical,
                'covid_positivity_rate': snapshot.covid_positivity_rate,
                'data_completeness': snapshot.data_completeness,
                'merge_confidence': snapshot.merge_confidence,
                'data_sources': ','.join(snapshot.data_sources_used),
            }

            # Add EHR fields
            if snapshot.ehr_data:
                record.update({
                    'total_patients': snapshot.ehr_data.total_patients,
                    'covid_positive': snapshot.ehr_data.covid_positive_count,
                    'covid_hospitalized': snapshot.ehr_data.covid_hospitalized,
                    'covid_icu': snapshot.ehr_data.covid_icu_count,
                    'covid_ventilator': snapshot.ehr_data.covid_ventilator_count,
                    'covid_deaths': snapshot.ehr_data.covid_deaths,
                    'tests_conducted': snapshot.ehr_data.tests_conducted,
                    'tests_positive': snapshot.ehr_data.tests_positive,
                    'vaccine_first_dose': snapshot.ehr_data.first_dose_administered,
                    'vaccine_second_dose': snapshot.ehr_data.second_dose_administered,
                })

            # Add staffing fields
            if snapshot.staffing_data:
                record.update({
                    'total_beds': snapshot.staffing_data.total_beds,
                    'occupied_beds': snapshot.staffing_data.occupied_beds,
                    'physicians_present': snapshot.staffing_data.physicians_present,
                    'nurses_present': snapshot.staffing_data.nurses_present,
                    'staff_covid_positive': snapshot.staffing_data.staff_covid_positive,
                    'staff_quarantined': snapshot.staffing_data.staff_quarantined,
                })

            # Add PPE fields
            if snapshot.ppe_data:
                record.update({
                    'n95_masks': snapshot.ppe_data.n95_masks_count,
                    'surgical_masks': snapshot.ppe_data.surgical_masks_count,
                    'gowns': snapshot.ppe_data.gowns_count,
                    'n95_days_supply': snapshot.ppe_data.n95_days_supply,
                    'ppe_critical_shortage': snapshot.ppe_data.critical_shortage,
                })

            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Exported {len(df)} records to DataFrame")
        return df

    def export_to_csv(self, output_path: str) -> None:
        """
        Export merged snapshots to CSV file.

        Args:
            output_path: Path to output CSV file
        """
        df = self.export_to_dataframe()
        df.to_csv(output_path, index=False)
        logger.info(f"Exported data to {output_path}")

    # ========================================================================
    # DATA QUALITY: Track and report issues
    # ========================================================================

    def _log_data_quality_issue(
        self,
        facility_id: str,
        issue_type: str,
        severity: str,
        description: str,
        affected_fields: List[str],
        issue_date: Optional[date] = None
    ) -> None:
        """
        Log a data quality issue.

        Args:
            facility_id: Facility identifier
            issue_type: Type of issue
            severity: Severity level
            description: Issue description
            affected_fields: List of affected field names
            issue_date: Date of issue (defaults to today)
        """
        if issue_date is None:
            issue_date = date.today()

        issue_id = hashlib.md5(
            f"{facility_id}_{issue_date}_{issue_type}_{datetime.now().timestamp()}".encode()
        ).hexdigest()[:16]

        issue = DataQualityIssue(
            issue_id=issue_id,
            facility_id=facility_id,
            issue_date=issue_date,
            issue_type=issue_type,
            severity=severity,
            description=description,
            affected_fields=affected_fields
        )

        self.data_quality_issues.append(issue)

    def get_data_quality_report(self) -> pd.DataFrame:
        """
        Generate data quality report.

        Returns:
            DataFrame with all data quality issues
        """
        if not self.data_quality_issues:
            return pd.DataFrame()

        records = [issue.dict() for issue in self.data_quality_issues]
        return pd.DataFrame(records)

    def get_deduplication_report(self) -> pd.DataFrame:
        """
        Generate deduplication report.

        Returns:
            DataFrame with all deduplication matches
        """
        if not self.deduplication_matches:
            return pd.DataFrame()

        records = [match.dict() for match in self.deduplication_matches]
        return pd.DataFrame(records)

    # ========================================================================
    # PIPELINE ORCHESTRATION
    # ========================================================================

    def run_pipeline(
        self,
        ehr_file: str,
        staffing_file: str,
        ppe_file: str,
        output_file: str
    ) -> Dict[str, any]:
        """
        Run complete ETL pipeline.

        Args:
            ehr_file: Path to EHR data CSV
            staffing_file: Path to staffing data CSV
            ppe_file: Path to PPE data CSV
            output_file: Path to output merged CSV

        Returns:
            Pipeline execution statistics
        """
        logger.info("=" * 80)
        logger.info("Starting COVID-19 Facility Data ETL Pipeline")
        logger.info("=" * 80)

        start_time = datetime.now()

        # EXTRACT
        logger.info("\n[1/4] EXTRACT: Loading data from source files...")
        ehr_count = self.extract_ehr_data(ehr_file)
        staffing_count = self.extract_staffing_data(staffing_file)
        ppe_count = self.extract_ppe_data(ppe_file)

        # TRANSFORM: Deduplication
        logger.info("\n[2/4] TRANSFORM: Deduplicating records...")
        dedup_stats = self.deduplicate_records()

        # TRANSFORM: Merge
        logger.info("\n[3/4] TRANSFORM: Merging data sources...")
        snapshot_count = self.merge_data_sources()

        # LOAD
        logger.info("\n[4/4] LOAD: Exporting merged data...")
        self.export_to_csv(output_file)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Generate statistics
        stats = {
            'pipeline_start': start_time,
            'pipeline_end': end_time,
            'duration_seconds': duration,
            'records_extracted': {
                'ehr': ehr_count,
                'staffing': staffing_count,
                'ppe': ppe_count,
                'total': ehr_count + staffing_count + ppe_count
            },
            'deduplication': dedup_stats,
            'snapshots_created': snapshot_count,
            'data_quality_issues': len(self.data_quality_issues),
            'output_file': output_file
        }

        logger.info("\n" + "=" * 80)
        logger.info("ETL Pipeline Complete!")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Records extracted: {stats['records_extracted']['total']:,}")
        logger.info(f"Duplicates removed: {dedup_stats['total_duplicates']:,}")
        logger.info(f"Snapshots created: {snapshot_count:,}")
        logger.info(f"Data quality issues: {len(self.data_quality_issues):,}")
        logger.info(f"Output: {output_file}")
        logger.info("=" * 80)

        return stats


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize ETL pipeline
    etl = FacilityDataETL(
        fuzzy_match_threshold=85,  # 85% similarity required for fuzzy matching
        date_tolerance_days=0       # Exact date matching required
    )

    # Run pipeline
    stats = etl.run_pipeline(
        ehr_file="data/ehr_data_20210115.csv",
        staffing_file="data/staffing_roster_20210115.csv",
        ppe_file="data/ppe_inventory_20210115.csv",
        output_file="output/merged_facility_data_20210115.csv"
    )

    # Export data quality reports
    dq_report = etl.get_data_quality_report()
    if not dq_report.empty:
        dq_report.to_csv("output/data_quality_issues_20210115.csv", index=False)
        print(f"\nData Quality Report: {len(dq_report)} issues logged")

    dedup_report = etl.get_deduplication_report()
    if not dedup_report.empty:
        dedup_report.to_csv("output/deduplication_matches_20210115.csv", index=False)
        print(f"Deduplication Report: {len(dedup_report)} matches found")
