"""
Data Ingestion Module for Multi-Stream Disease Surveillance
Handles data loading and preprocessing from ESSENCE, wastewater, search trends, and mobility sources
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
from abc import ABC, abstractmethod

from .models import (
    ESSENCERecord,
    WastewaterRecord,
    SearchTrendsRecord,
    MobilityRecord,
    GeographicLocation,
    DataStreamType
)


class DataStreamIngester(ABC):
    """Base class for data stream ingesters"""

    def __init__(self, data_source_name: str):
        self.data_source_name = data_source_name

    @abstractmethod
    def ingest(self, data_path: Path) -> pd.DataFrame:
        """Ingest data from source"""
        pass

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate ingested data"""
        pass

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Common preprocessing steps"""
        # Remove duplicates
        df = df.drop_duplicates()

        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')

        # Reset index
        df = df.reset_index(drop=True)

        return df


class ESSENCEIngester(DataStreamIngester):
    """
    Ingester for ESSENCE syndromic surveillance data

    Expected format:
    - Date/Time
    - Region/County
    - Chief Complaint Category
    - Visit Count
    - Total Visits
    - Demographics (optional)
    """

    def __init__(self):
        super().__init__("ESSENCE")

    def ingest(self, data_path: Path) -> pd.DataFrame:
        """
        Ingest ESSENCE data from CSV or Excel

        Args:
            data_path: Path to data file

        Returns:
            DataFrame with standardized columns
        """
        # Try to read as CSV first, then Excel
        try:
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
            elif data_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")

        except Exception as e:
            raise RuntimeError(f"Failed to read ESSENCE data: {str(e)}")

        # Standardize column names
        df = self._standardize_columns(df)

        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Calculate percentage if not present
        if 'percentage' not in df.columns:
            df['percentage'] = (df['visit_count'] / df['total_visits'] * 100).fillna(0)

        return self.preprocess(df)

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to expected format"""
        column_mapping = {
            'date': 'timestamp',
            'datetime': 'timestamp',
            'date_time': 'timestamp',
            'region': 'region_name',
            'county': 'region_name',
            'location': 'region_name',
            'chief_complaint': 'chief_complaint_category',
            'complaint_category': 'chief_complaint_category',
            'category': 'chief_complaint_category',
            'visits': 'visit_count',
            'count': 'visit_count',
            'total': 'total_visits',
            'all_visits': 'total_visits',
            'age': 'age_group',
            'gender': 'sex'
        }

        df.columns = df.columns.str.lower().str.strip()

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})

        return df

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate ESSENCE data"""
        errors = []

        required_columns = ['timestamp', 'region_name', 'chief_complaint_category', 'visit_count']

        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                errors.append("timestamp column must be datetime type")

        if 'visit_count' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['visit_count']):
                errors.append("visit_count must be numeric")

        return len(errors) == 0, errors

    def convert_to_records(self, df: pd.DataFrame) -> List[ESSENCERecord]:
        """Convert DataFrame to ESSENCERecord objects"""
        records = []

        for _, row in df.iterrows():
            # Create location
            location = GeographicLocation(
                latitude=row.get('latitude', 0.0),
                longitude=row.get('longitude', 0.0),
                region_name=row['region_name'],
                region_code=row.get('region_code', row['region_name']),
                population=row.get('population')
            )

            # Create record
            record = ESSENCERecord(
                timestamp=row['timestamp'],
                location=location,
                chief_complaint_category=row['chief_complaint_category'],
                visit_count=int(row['visit_count']),
                total_visits=int(row.get('total_visits', row['visit_count'])),
                percentage=float(row.get('percentage', 0.0)),
                age_group=row.get('age_group'),
                sex=row.get('sex')
            )
            records.append(record)

        return records


class WastewaterIngester(DataStreamIngester):
    """
    Ingester for wastewater viral load data

    Expected format:
    - Date
    - Collection Site
    - Viral Load (copies/L)
    - Pathogen
    - Population Served
    """

    def __init__(self):
        super().__init__("Wastewater")

    def ingest(self, data_path: Path) -> pd.DataFrame:
        """Ingest wastewater data"""
        try:
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
            elif data_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")

        except Exception as e:
            raise RuntimeError(f"Failed to read wastewater data: {str(e)}")

        df = self._standardize_columns(df)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        return self.preprocess(df)

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        column_mapping = {
            'date': 'timestamp',
            'collection_date': 'timestamp',
            'sample_date': 'timestamp',
            'site': 'collection_site_id',
            'site_id': 'collection_site_id',
            'location': 'collection_site_id',
            'viral_concentration': 'viral_load',
            'concentration': 'viral_load',
            'copies_per_liter': 'viral_load',
            'virus': 'pathogen',
            'organism': 'pathogen',
            'population': 'population_served'
        }

        df.columns = df.columns.str.lower().str.strip()

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})

        return df

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate wastewater data"""
        errors = []

        required_columns = ['timestamp', 'collection_site_id', 'viral_load', 'pathogen']

        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        if 'viral_load' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['viral_load']):
                errors.append("viral_load must be numeric")
            elif (df['viral_load'] < 0).any():
                errors.append("viral_load cannot be negative")

        return len(errors) == 0, errors

    def convert_to_records(self, df: pd.DataFrame) -> List[WastewaterRecord]:
        """Convert DataFrame to WastewaterRecord objects"""
        records = []

        for _, row in df.iterrows():
            location = GeographicLocation(
                latitude=row.get('latitude', 0.0),
                longitude=row.get('longitude', 0.0),
                region_name=row.get('region_name', row['collection_site_id']),
                region_code=row.get('region_code', row['collection_site_id']),
                population=row.get('population_served')
            )

            record = WastewaterRecord(
                timestamp=row['timestamp'],
                location=location,
                viral_load=float(row['viral_load']),
                pathogen=row['pathogen'],
                collection_site_id=row['collection_site_id'],
                flow_rate=row.get('flow_rate'),
                population_served=row.get('population_served')
            )
            records.append(record)

        return records


class SearchTrendsIngester(DataStreamIngester):
    """
    Ingester for Google/Search Trends data

    Expected format:
    - Date
    - Region
    - Search Term
    - Normalized Interest (0-100)
    """

    def __init__(self):
        super().__init__("SearchTrends")

    def ingest(self, data_path: Path) -> pd.DataFrame:
        """Ingest search trends data"""
        try:
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
            elif data_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(data_path)
            elif data_path.suffix.lower() == '.json':
                df = pd.read_json(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")

        except Exception as e:
            raise RuntimeError(f"Failed to read search trends data: {str(e)}")

        df = self._standardize_columns(df)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        return self.preprocess(df)

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        column_mapping = {
            'date': 'timestamp',
            'week': 'timestamp',
            'region': 'region_name',
            'location': 'region_name',
            'geoname': 'region_name',
            'term': 'search_term',
            'query': 'search_term',
            'keyword': 'search_term',
            'interest': 'normalized_interest',
            'value': 'normalized_interest',
            'score': 'normalized_interest'
        }

        df.columns = df.columns.str.lower().str.strip()

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})

        return df

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate search trends data"""
        errors = []

        required_columns = ['timestamp', 'region_name', 'search_term', 'normalized_interest']

        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        if 'normalized_interest' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['normalized_interest']):
                errors.append("normalized_interest must be numeric")
            elif ((df['normalized_interest'] < 0) | (df['normalized_interest'] > 100)).any():
                errors.append("normalized_interest must be between 0 and 100")

        return len(errors) == 0, errors

    def convert_to_records(self, df: pd.DataFrame) -> List[SearchTrendsRecord]:
        """Convert DataFrame to SearchTrendsRecord objects"""
        records = []

        for _, row in df.iterrows():
            location = GeographicLocation(
                latitude=row.get('latitude', 0.0),
                longitude=row.get('longitude', 0.0),
                region_name=row['region_name'],
                region_code=row.get('region_code', row['region_name'])
            )

            record = SearchTrendsRecord(
                timestamp=row['timestamp'],
                location=location,
                search_term=row['search_term'],
                normalized_interest=float(row['normalized_interest']),
                related_queries=row.get('related_queries', [])
            )
            records.append(record)

        return records


class MobilityIngester(DataStreamIngester):
    """
    Ingester for mobility/movement data

    Expected format:
    - Date
    - Region
    - Category (retail, transit, workplaces, etc.)
    - Baseline Comparison (% change)
    """

    def __init__(self):
        super().__init__("Mobility")

    def ingest(self, data_path: Path) -> pd.DataFrame:
        """Ingest mobility data"""
        try:
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
            elif data_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")

        except Exception as e:
            raise RuntimeError(f"Failed to read mobility data: {str(e)}")

        df = self._standardize_columns(df)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        return self.preprocess(df)

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        column_mapping = {
            'date': 'timestamp',
            'region': 'region_name',
            'location': 'region_name',
            'type': 'category',
            'place_category': 'category',
            'change_from_baseline': 'baseline_comparison',
            'percent_change': 'baseline_comparison',
            'mobility_change': 'baseline_comparison',
            'index': 'mobility_index'
        }

        df.columns = df.columns.str.lower().str.strip()

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})

        return df

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate mobility data"""
        errors = []

        required_columns = ['timestamp', 'region_name', 'category', 'baseline_comparison']

        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        if 'baseline_comparison' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['baseline_comparison']):
                errors.append("baseline_comparison must be numeric")

        return len(errors) == 0, errors

    def convert_to_records(self, df: pd.DataFrame) -> List[MobilityRecord]:
        """Convert DataFrame to MobilityRecord objects"""
        records = []

        for _, row in df.iterrows():
            location = GeographicLocation(
                latitude=row.get('latitude', 0.0),
                longitude=row.get('longitude', 0.0),
                region_name=row['region_name'],
                region_code=row.get('region_code', row['region_name'])
            )

            record = MobilityRecord(
                timestamp=row['timestamp'],
                location=location,
                mobility_index=row.get('mobility_index', 100 + row['baseline_comparison']),
                baseline_comparison=float(row['baseline_comparison']),
                category=row['category']
            )
            records.append(record)

        return records


class MultiStreamDataIngester:
    """
    Coordinator for ingesting multiple data streams
    """

    def __init__(self):
        self.essence_ingester = ESSENCEIngester()
        self.wastewater_ingester = WastewaterIngester()
        self.search_trends_ingester = SearchTrendsIngester()
        self.mobility_ingester = MobilityIngester()

    def ingest_all(
        self,
        essence_path: Optional[Path] = None,
        wastewater_path: Optional[Path] = None,
        search_trends_path: Optional[Path] = None,
        mobility_path: Optional[Path] = None
    ) -> Dict[DataStreamType, pd.DataFrame]:
        """
        Ingest all available data streams

        Returns:
            Dictionary mapping stream type to DataFrame
        """
        data_streams = {}

        if essence_path and essence_path.exists():
            try:
                df = self.essence_ingester.ingest(essence_path)
                valid, errors = self.essence_ingester.validate(df)
                if valid:
                    data_streams[DataStreamType.ESSENCE_SYNDROMIC] = df
                else:
                    warnings.warn(f"ESSENCE validation failed: {errors}")
            except Exception as e:
                warnings.warn(f"Failed to ingest ESSENCE data: {str(e)}")

        if wastewater_path and wastewater_path.exists():
            try:
                df = self.wastewater_ingester.ingest(wastewater_path)
                valid, errors = self.wastewater_ingester.validate(df)
                if valid:
                    data_streams[DataStreamType.WASTEWATER] = df
                else:
                    warnings.warn(f"Wastewater validation failed: {errors}")
            except Exception as e:
                warnings.warn(f"Failed to ingest wastewater data: {str(e)}")

        if search_trends_path and search_trends_path.exists():
            try:
                df = self.search_trends_ingester.ingest(search_trends_path)
                valid, errors = self.search_trends_ingester.validate(df)
                if valid:
                    data_streams[DataStreamType.SEARCH_TRENDS] = df
                else:
                    warnings.warn(f"Search trends validation failed: {errors}")
            except Exception as e:
                warnings.warn(f"Failed to ingest search trends data: {str(e)}")

        if mobility_path and mobility_path.exists():
            try:
                df = self.mobility_ingester.ingest(mobility_path)
                valid, errors = self.mobility_ingester.validate(df)
                if valid:
                    data_streams[DataStreamType.MOBILITY] = df
                else:
                    warnings.warn(f"Mobility validation failed: {errors}")
            except Exception as e:
                warnings.warn(f"Failed to ingest mobility data: {str(e)}")

        return data_streams
