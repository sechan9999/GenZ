#!/usr/bin/env python3
"""
Sample Data Generator for COVID-19 Facility Analysis Pipeline

Generates realistic sample data for testing the analysis pipeline:
- EHR data (Electronic Health Records)
- Staffing rosters
- PPE inventory logs

Usage:
    python generate_sample_data.py --facilities 10 --days 30
    python generate_sample_data.py --output data/ --facilities 170 --days 7
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random


class SampleDataGenerator:
    """Generate realistic sample COVID-19 facility data."""

    def __init__(self, num_facilities: int = 170, num_days: int = 30, seed: int = 42):
        """
        Initialize sample data generator.

        Args:
            num_facilities: Number of facilities to generate
            num_days: Number of days of data
            seed: Random seed for reproducibility
        """
        self.num_facilities = num_facilities
        self.num_days = num_days

        np.random.seed(seed)
        random.seed(seed)

        # Generate facility definitions
        self.facilities = self._generate_facilities()

    def _generate_facilities(self):
        """Generate facility definitions."""
        facility_types = ['medical_center', 'outpatient_clinic', 'long_term_care', 'community_living_center']
        facility_weights = [0.1, 0.2, 0.6, 0.1]  # Most are long-term care

        states = ['NY', 'CA', 'TX', 'FL', 'PA', 'IL', 'OH', 'NC', 'MI', 'VA', 'GA', 'WA']
        cities = ['Buffalo', 'Albany', 'Los Angeles', 'San Diego', 'Houston', 'Dallas',
                 'Miami', 'Tampa', 'Philadelphia', 'Pittsburgh', 'Chicago', 'Columbus']

        facilities = []
        for i in range(self.num_facilities):
            facility_id = f"VA-{500 + i}"
            facility_type = np.random.choice(facility_types, p=facility_weights)

            state = random.choice(states)
            city = random.choice(cities)

            if facility_type == 'medical_center':
                name = f"VA {city} Healthcare System"
            elif facility_type == 'outpatient_clinic':
                name = f"VA {city} Outpatient Clinic"
            elif facility_type == 'long_term_care':
                name = f"VA {city} Long-Term Care Facility"
            else:
                name = f"VA {city} Community Living Center"

            # Facility characteristics
            if facility_type == 'medical_center':
                bed_capacity = np.random.randint(200, 500)
                base_patients = np.random.randint(800, 2000)
            elif facility_type == 'outpatient_clinic':
                bed_capacity = np.random.randint(20, 50)
                base_patients = np.random.randint(100, 300)
            elif facility_type == 'long_term_care':
                bed_capacity = np.random.randint(50, 150)
                base_patients = np.random.randint(40, 120)
            else:
                bed_capacity = np.random.randint(30, 100)
                base_patients = np.random.randint(25, 80)

            facilities.append({
                'facility_id': facility_id,
                'facility_name': name,
                'facility_type': facility_type,
                'bed_capacity': bed_capacity,
                'base_patients': base_patients,
                'state': state,
                'city': city
            })

        return facilities

    def generate_ehr_data(self, output_path: str):
        """Generate EHR (Electronic Health Records) data."""
        print(f"Generating EHR data for {self.num_facilities} facilities over {self.num_days} days...")

        records = []
        end_date = datetime.now().date()

        for facility in self.facilities:
            # Simulate a COVID outbreak curve for this facility
            outbreak_peak = np.random.randint(10, self.num_days)
            outbreak_severity = np.random.uniform(0.02, 0.15)  # 2-15% of patients

            for day in range(self.num_days):
                record_date = end_date - timedelta(days=self.num_days - day - 1)

                # Base patient count with random variation
                total_patients = int(facility['base_patients'] * np.random.uniform(0.9, 1.1))

                # COVID outbreak curve (Gaussian-ish)
                outbreak_factor = np.exp(-((day - outbreak_peak) ** 2) / (2 * (self.num_days / 4) ** 2))
                covid_rate = outbreak_severity * outbreak_factor

                covid_positive = int(total_patients * covid_rate)
                covid_positive = max(0, covid_positive + np.random.randint(-5, 5))

                # Derived metrics
                covid_hospitalized = int(covid_positive * np.random.uniform(0.15, 0.35))
                covid_icu = int(covid_hospitalized * np.random.uniform(0.15, 0.30))
                covid_ventilator = int(covid_icu * np.random.uniform(0.40, 0.70))
                covid_deaths = int(covid_positive * np.random.uniform(0.001, 0.02))

                # Testing
                tests_conducted = int(total_patients * np.random.uniform(0.10, 0.30))
                tests_positive = min(covid_positive, tests_conducted)
                tests_pending = np.random.randint(0, int(tests_conducted * 0.05))

                # Vaccination
                first_dose = int(total_patients * np.random.uniform(0.20, 0.50))
                second_dose = int(first_dose * np.random.uniform(0.60, 0.90))
                booster = int(second_dose * np.random.uniform(0.10, 0.40))

                record = {
                    'record_id': f"EHR-{record_date.strftime('%Y%m%d')}-{facility['facility_id']}-001",
                    'facility_id': facility['facility_id'],
                    'facility_name': facility['facility_name'],
                    'facility_type': facility['facility_type'],
                    'record_date': record_date,
                    'total_patients': total_patients,
                    'covid_positive_count': covid_positive,
                    'covid_hospitalized': covid_hospitalized,
                    'covid_icu_count': covid_icu,
                    'covid_ventilator_count': covid_ventilator,
                    'covid_deaths': covid_deaths,
                    'tests_conducted': tests_conducted,
                    'tests_positive': tests_positive,
                    'tests_pending': tests_pending,
                    'first_dose_administered': first_dose,
                    'second_dose_administered': second_dose,
                    'booster_dose_administered': booster,
                }
                records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        print(f"✅ EHR data saved: {output_path} ({len(df):,} records)")
        return df

    def generate_staffing_data(self, output_path: str):
        """Generate staffing roster data."""
        print(f"Generating staffing data for {self.num_facilities} facilities over {self.num_days} days...")

        records = []
        end_date = datetime.now().date()

        for facility in self.facilities:
            bed_capacity = facility['bed_capacity']

            # Staffing levels based on facility size
            if facility['facility_type'] == 'medical_center':
                base_physicians = np.random.randint(40, 60)
                base_nurses = np.random.randint(150, 200)
                base_rt = np.random.randint(10, 20)
                base_support = np.random.randint(80, 120)
            elif facility['facility_type'] == 'outpatient_clinic':
                base_physicians = np.random.randint(5, 15)
                base_nurses = np.random.randint(10, 25)
                base_rt = np.random.randint(0, 3)
                base_support = np.random.randint(5, 15)
            else:
                base_physicians = np.random.randint(3, 10)
                base_nurses = np.random.randint(20, 40)
                base_rt = np.random.randint(1, 5)
                base_support = np.random.randint(10, 25)

            for day in range(self.num_days):
                roster_date = end_date - timedelta(days=self.num_days - day - 1)

                # Attendance rate (slightly lower on weekends)
                is_weekend = roster_date.weekday() >= 5
                attendance_rate = np.random.uniform(0.85, 0.95) if not is_weekend else np.random.uniform(0.70, 0.85)

                physicians_scheduled = base_physicians
                physicians_present = int(base_physicians * attendance_rate)

                nurses_scheduled = base_nurses
                nurses_present = int(base_nurses * attendance_rate)

                rt_scheduled = base_rt
                rt_present = int(base_rt * attendance_rate)

                support_scheduled = base_support
                support_present = int(base_support * attendance_rate)

                # Staff health
                total_staff = physicians_scheduled + nurses_scheduled + rt_scheduled + support_scheduled
                staff_covid_positive = np.random.binomial(total_staff, 0.01)  # 1% positive rate
                staff_quarantined = np.random.binomial(total_staff, 0.02)  # 2% quarantine rate
                staff_vaccinated_full = int(total_staff * np.random.uniform(0.70, 0.95))

                # Bed utilization
                occupied_beds = int(bed_capacity * np.random.uniform(0.60, 0.90))
                covid_beds_available = np.random.randint(5, 25)

                icu_beds_total = int(bed_capacity * 0.10)
                icu_beds_occupied = int(icu_beds_total * np.random.uniform(0.60, 0.95))

                record = {
                    'roster_id': f"ROSTER-{roster_date.strftime('%Y%m%d')}-{facility['facility_id']}",
                    'facility_id': facility['facility_id'],
                    'facility_name': facility['facility_name'],
                    'roster_date': roster_date,
                    'physicians_scheduled': physicians_scheduled,
                    'physicians_present': physicians_present,
                    'nurses_scheduled': nurses_scheduled,
                    'nurses_present': nurses_present,
                    'respiratory_therapists_scheduled': rt_scheduled,
                    'respiratory_therapists_present': rt_present,
                    'support_staff_scheduled': support_scheduled,
                    'support_staff_present': support_present,
                    'staff_covid_positive': staff_covid_positive,
                    'staff_quarantined': staff_quarantined,
                    'staff_vaccinated_full': staff_vaccinated_full,
                    'total_beds': bed_capacity,
                    'occupied_beds': occupied_beds,
                    'covid_beds_available': covid_beds_available,
                    'icu_beds_total': icu_beds_total,
                    'icu_beds_occupied': icu_beds_occupied,
                }
                records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        print(f"✅ Staffing data saved: {output_path} ({len(df):,} records)")
        return df

    def generate_ppe_data(self, output_path: str):
        """Generate PPE inventory data."""
        print(f"Generating PPE data for {self.num_facilities} facilities over {self.num_days} days...")

        records = []
        end_date = datetime.now().date()

        for facility in self.facilities:
            # Initial inventory levels
            n95_stock = np.random.randint(2000, 10000)
            surgical_mask_stock = np.random.randint(5000, 20000)
            face_shields = np.random.randint(500, 2000)
            gowns = np.random.randint(1000, 5000)
            gloves_boxes = np.random.randint(200, 800)

            for day in range(self.num_days):
                inventory_date = end_date - timedelta(days=self.num_days - day - 1)

                # Simulate daily usage and occasional restocking
                daily_usage_rate = 0.05  # 5% per day
                restock_chance = 0.15  # 15% chance of restocking each day

                if np.random.random() < restock_chance:
                    # Restock
                    n95_stock += np.random.randint(1000, 3000)
                    surgical_mask_stock += np.random.randint(2000, 5000)
                    face_shields += np.random.randint(200, 500)
                    gowns += np.random.randint(500, 1500)
                    gloves_boxes += np.random.randint(100, 300)
                else:
                    # Consume
                    n95_stock = max(0, int(n95_stock * (1 - daily_usage_rate)))
                    surgical_mask_stock = max(0, int(surgical_mask_stock * (1 - daily_usage_rate)))
                    face_shields = max(0, int(face_shields * (1 - daily_usage_rate)))
                    gowns = max(0, int(gowns * (1 - daily_usage_rate)))
                    gloves_boxes = max(0, int(gloves_boxes * (1 - daily_usage_rate)))

                # Calculate days of supply
                n95_days_supply = n95_stock / max(1, n95_stock * daily_usage_rate)
                surgical_days_supply = surgical_mask_stock / max(1, surgical_mask_stock * daily_usage_rate)
                gown_days_supply = gowns / max(1, gowns * daily_usage_rate)

                # Critical shortage if < 7 days supply
                critical_shortage = (n95_days_supply < 7 or surgical_days_supply < 7 or gown_days_supply < 7)

                # Reorder list
                reorder_needed = []
                if n95_days_supply < 10:
                    reorder_needed.append('n95_masks')
                if surgical_days_supply < 10:
                    reorder_needed.append('surgical_masks')
                if gown_days_supply < 10:
                    reorder_needed.append('gowns')
                if face_shields < 300:
                    reorder_needed.append('face_shields')

                # Ventilators
                ventilators_total = np.random.randint(10, 30)
                ventilators_in_use = np.random.randint(5, ventilators_total)

                record = {
                    'inventory_id': f"PPE-{inventory_date.strftime('%Y%m%d')}-{facility['facility_id']}",
                    'facility_id': facility['facility_id'],
                    'facility_name': facility['facility_name'],
                    'inventory_date': inventory_date,
                    'n95_masks_count': n95_stock,
                    'surgical_masks_count': surgical_mask_stock,
                    'face_shields_count': face_shields,
                    'gowns_count': gowns,
                    'gloves_boxes': gloves_boxes,
                    'hand_sanitizer_bottles': np.random.randint(100, 500),
                    'disinfectant_wipes_count': np.random.randint(100, 400),
                    'ventilators_total': ventilators_total,
                    'ventilators_in_use': ventilators_in_use,
                    'n95_days_supply': round(n95_days_supply, 1),
                    'surgical_mask_days_supply': round(surgical_days_supply, 1),
                    'gown_days_supply': round(gown_days_supply, 1),
                    'critical_shortage': critical_shortage,
                    'reorder_needed': ','.join(reorder_needed) if reorder_needed else '',
                }
                records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        print(f"✅ PPE data saved: {output_path} ({len(df):,} records)")
        return df

    def generate_all(self, output_dir: str = "data"):
        """Generate all sample data files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")

        print("\n" + "=" * 80)
        print("SAMPLE DATA GENERATION")
        print("=" * 80)
        print(f"Facilities: {self.num_facilities}")
        print(f"Days: {self.num_days}")
        print(f"Total records: ~{self.num_facilities * self.num_days * 3:,}")
        print("=" * 80 + "\n")

        # Generate EHR data
        ehr_path = output_path / f"ehr_data_{timestamp}.csv"
        self.generate_ehr_data(str(ehr_path))

        # Generate staffing data
        staffing_path = output_path / f"staffing_roster_{timestamp}.csv"
        self.generate_staffing_data(str(staffing_path))

        # Generate PPE data
        ppe_path = output_path / f"ppe_inventory_{timestamp}.csv"
        self.generate_ppe_data(str(ppe_path))

        print("\n" + "=" * 80)
        print("✅ ALL SAMPLE DATA GENERATED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nTo run analysis pipeline:")
        print(f"  python main.py \\")
        print(f"    --ehr {ehr_path} \\")
        print(f"    --staffing {staffing_path} \\")
        print(f"    --ppe {ppe_path} \\")
        print(f"    --output output/")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate sample COVID-19 facility data for testing'
    )

    parser.add_argument(
        '--facilities',
        type=int,
        default=170,
        help='Number of facilities to generate (default: 170)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days of data (default: 30)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='Output directory (default: data/)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    generator = SampleDataGenerator(
        num_facilities=args.facilities,
        num_days=args.days,
        seed=args.seed
    )

    generator.generate_all(output_dir=args.output)


if __name__ == "__main__":
    main()
