"""
LIMS 실무 KPI 대시보드 - 바로 실행 가능한 풀 패키지

실제 병원/검사실에서 매일 확인하는 핵심 지표:
- TAT (Turn Around Time): 검사 소요 시간
- COVID-19 양성률 추이
- 일별 검사량
- 지역별 양성률

Author: MLOps LIMS Team
Date: 2025-11-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# 한글 폰트 설정 (Windows/Mac/Linux 호환)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 1. 실제 LIMS에서 나오는 원시 데이터 (실제 컬럼명 그대로)
def generate_sample_lims_data():
    """
    실제 LIMS 시스템에서 추출하는 형태의 샘플 데이터 생성

    실제 LIMS 테이블 구조:
    - SPECIMEN_ID: 검체 식별자
    - PATIENT_ID: 환자 식별자 (해싱 처리 필요)
    - ORDER_DATE: 검사 오더 일시
    - COLLECT_DATE: 검체 채취 일시
    - RECEIVE_DATE: 검사실 접수 일시
    - RESULT_DATE: 결과 보고 일시
    - TEST_NAME: 검사명
    - RESULT: 검사 결과
    - CT_VALUE: Ct 값 (COVID PCR 검사)
    - LAB_LOCATION: 검사실 위치
    """
    raw_lims = pd.DataFrame({
        'SPECIMEN_ID': ['S2025001', 'S2025002', 'S2025003', 'S2025004', 'S2025005', 'S2025006'],
        'PATIENT_ID': ['P1001', 'P1002', 'P1001', 'P1003', 'P1004', 'P1005'],
        'ORDER_DATE': [
            '2025-01-10 08:30', '2025-01-10 09:15', '2025-01-10 14:20',
            '2025-01-11 07:45', '2025-01-11 10:30', '2025-01-11 16:00'
        ],
        'COLLECT_DATE': [
            '2025-01-10 09:00', '2025-01-10 09:45', '2025-01-10 15:00',
            '2025-01-11 08:15', '2025-01-11 11:00', '2025-01-11 16:30'
        ],
        'RECEIVE_DATE': [
            '2025-01-10 10:30', '2025-01-10 11:00', '2025-01-10 16:30',
            '2025-01-11 09:45', '2025-01-11 12:30', '2025-01-11 18:00'
        ],
        'RESULT_DATE': [
            '2025-01-10 14:20', '2025-01-10 15:45', '2025-01-11 09:15',
            '2025-01-11 13:30', '2025-01-11 18:20', '2025-01-12 10:30'
        ],
        'TEST_NAME': [
            'COVID-19 PCR', 'COVID-19 PCR', 'Influenza A/B',
            'COVID-19 PCR', 'CBC', 'COVID-19 PCR'
        ],
        'RESULT': [
            'Detected', 'Not Detected', 'Influenza A Detected',
            'Detected', 'WBC 12.5', 'Not Detected'
        ],
        'CT_VALUE': [22.5, np.nan, np.nan, 35.8, np.nan, np.nan],
        'LAB_LOCATION': ['Seoul', 'Seoul', 'Busan', 'Seoul', 'Daegu', 'Seoul']
    })

    # 날짜형 변환
    date_cols = ['ORDER_DATE', 'COLLECT_DATE', 'RECEIVE_DATE', 'RESULT_DATE']
    for col in date_cols:
        raw_lims[col] = pd.to_datetime(raw_lims[col])

    return raw_lims


def lims_kpi_dashboard(df, show_plots=True):
    """
    LIMS 실무 KPI 대시보드

    실무에서 매일 아침 확인하는 핵심 지표:
    1. 총 검사 건수
    2. COVID-19 검사 건수 및 양성률
    3. TAT (Turn Around Time) 분석
    4. 지연 케이스 식별

    Args:
        df: LIMS 원시 데이터 DataFrame
        show_plots: 시각화 표시 여부

    Returns:
        dict: KPI 메트릭
    """
    # TAT (Turn Around Time) 계산
    df = df.copy()
    df['ORDER_TO_COLLECT'] = (df['COLLECT_DATE'] - df['ORDER_DATE']).dt.total_seconds() / 3600
    df['COLLECT_TO_RECEIVE'] = (df['RECEIVE_DATE'] - df['COLLECT_DATE']).dt.total_seconds() / 3600
    df['RECEIVE_TO_RESULT'] = (df['RESULT_DATE'] - df['RECEIVE_DATE']).dt.total_seconds() / 3600
    df['TOTAL_TAT'] = (df['RESULT_DATE'] - df['ORDER_DATE']).dt.total_seconds() / 3600

    # COVID-19 검사만 필터링
    covid = df[df['TEST_NAME'].str.contains('COVID', na=False)].copy()
    covid['IS_POSITIVE'] = covid['RESULT'].str.contains('Detected', na=False)

    # KPI 계산
    kpi_metrics = {
        'total_tests': len(df),
        'covid_tests': len(covid),
        'covid_positive_rate': covid['IS_POSITIVE'].mean() if len(covid) > 0 else 0,
        'avg_total_tat': df['TOTAL_TAT'].mean(),
        'avg_covid_tat': covid['TOTAL_TAT'].mean() if len(covid) > 0 else 0,
        'max_tat': df['TOTAL_TAT'].max(),
        'max_tat_specimen': df.loc[df['TOTAL_TAT'].idxmax(), 'SPECIMEN_ID'] if len(df) > 0 else None
    }

    # 콘솔 출력
    print("=" * 70)
    print("LIMS Daily KPI Dashboard")
    print("=" * 70)
    print(f"Period: {df['ORDER_DATE'].min().date()} ~ {df['ORDER_DATE'].max().date()}")
    print("-" * 70)
    print(f"Total Tests             : {kpi_metrics['total_tests']:,} tests")
    print(f"COVID-19 Tests          : {kpi_metrics['covid_tests']:,} tests")
    print(f"COVID Positive Rate     : {kpi_metrics['covid_positive_rate']:.1%}")
    print(f"Avg Total TAT           : {kpi_metrics['avg_total_tat']:.1f} hours")
    print(f"Avg COVID TAT           : {kpi_metrics['avg_covid_tat']:.1f} hours")
    print(f"Max TAT (Alert Case)    : {kpi_metrics['max_tat']:.1f} hours")
    print(f"Max TAT Specimen ID     : {kpi_metrics['max_tat_specimen']}")
    print("=" * 70)

    # 시각화
    if show_plots:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('LIMS Daily KPI Dashboard', fontsize=16, fontweight='bold')

        # 1. Daily Test Volume
        daily_tests = df['ORDER_DATE'].dt.date.value_counts().sort_index()
        axes[0, 0].bar(range(len(daily_tests)), daily_tests.values, color='skyblue', alpha=0.7)
        axes[0, 0].set_xticks(range(len(daily_tests)))
        axes[0, 0].set_xticklabels([str(d) for d in daily_tests.index], rotation=45)
        axes[0, 0].set_title('Daily Test Volume', fontweight='bold')
        axes[0, 0].set_ylabel('Number of Tests')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. TAT Distribution
        axes[0, 1].hist(df['TOTAL_TAT'], bins=30, color='orange', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(
            df['TOTAL_TAT'].mean(),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {df["TOTAL_TAT"].mean():.1f}h'
        )
        axes[0, 1].set_title('Total TAT Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('TAT (hours)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 3. COVID Positive Rate Trend
        if len(covid) > 0:
            covid_daily = covid.groupby(covid['ORDER_DATE'].dt.date)['IS_POSITIVE'].mean()
            axes[1, 0].plot(
                range(len(covid_daily)),
                covid_daily.values * 100,
                marker='o',
                color='red',
                linewidth=2,
                markersize=8
            )
            axes[1, 0].set_xticks(range(len(covid_daily)))
            axes[1, 0].set_xticklabels([str(d) for d in covid_daily.index], rotation=45)
            axes[1, 0].set_title('Daily COVID Positive Rate', fontweight='bold')
            axes[1, 0].set_ylabel('Positive Rate (%)')
            axes[1, 0].grid(alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No COVID tests', ha='center', va='center')
            axes[1, 0].set_title('Daily COVID Positive Rate', fontweight='bold')

        # 4. Regional COVID Positive Rate
        if len(covid) > 0:
            covid['REGION'] = covid['LAB_LOCATION']
            regional_rate = covid.groupby('REGION')['IS_POSITIVE'].mean().sort_values()
            axes[1, 1].barh(range(len(regional_rate)), regional_rate.values * 100, color='purple', alpha=0.7)
            axes[1, 1].set_yticks(range(len(regional_rate)))
            axes[1, 1].set_yticklabels(regional_rate.index)
            axes[1, 1].set_title('Regional COVID Positive Rate', fontweight='bold')
            axes[1, 1].set_xlabel('Positive Rate (%)')
            axes[1, 1].grid(axis='x', alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No COVID tests', ha='center', va='center')
            axes[1, 1].set_title('Regional COVID Positive Rate', fontweight='bold')

        plt.tight_layout()
        plt.savefig('lims_kpi_dashboard.png', dpi=150, bbox_inches='tight')
        print("\n✓ Dashboard saved to: lims_kpi_dashboard.png")
        plt.show()

    return kpi_metrics


def detect_delayed_cases(df, tat_threshold=24):
    """
    TAT 임계값을 초과하는 지연 케이스 탐지

    Args:
        df: LIMS 데이터
        tat_threshold: TAT 임계값 (시간 단위)

    Returns:
        DataFrame: 지연 케이스 목록
    """
    df = df.copy()
    df['TOTAL_TAT'] = (df['RESULT_DATE'] - df['ORDER_DATE']).dt.total_seconds() / 3600

    delayed = df[df['TOTAL_TAT'] > tat_threshold].copy()
    delayed = delayed.sort_values('TOTAL_TAT', ascending=False)

    print(f"\nDelayed Cases (TAT > {tat_threshold} hours):")
    print("=" * 70)
    if len(delayed) > 0:
        for idx, row in delayed.iterrows():
            print(f"Specimen ID: {row['SPECIMEN_ID']}")
            print(f"  Test: {row['TEST_NAME']}")
            print(f"  TAT: {row['TOTAL_TAT']:.1f} hours")
            print(f"  Order Date: {row['ORDER_DATE']}")
            print(f"  Result Date: {row['RESULT_DATE']}")
            print("-" * 70)
    else:
        print("No delayed cases found.")

    return delayed


def export_daily_report(df, output_path='lims_daily_report.csv'):
    """
    일일 리포트를 CSV로 저장 (Excel은 openpyxl 설치 필요)

    Args:
        df: LIMS 데이터
        output_path: 출력 파일 경로
    """
    df = df.copy()
    df['TOTAL_TAT'] = (df['RESULT_DATE'] - df['ORDER_DATE']).dt.total_seconds() / 3600

    # CSV 파일로 저장 (openpyxl 없이도 작동)
    try:
        # openpyxl이 있으면 Excel로 저장
        import openpyxl
        excel_path = output_path.replace('.csv', '.xlsx')

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: 전체 데이터
            df.to_excel(writer, sheet_name='Raw Data', index=False)

            # Sheet 2: 일별 요약
            daily_summary = df.groupby(df['ORDER_DATE'].dt.date).agg({
                'SPECIMEN_ID': 'count',
                'TOTAL_TAT': 'mean'
            }).rename(columns={'SPECIMEN_ID': 'Test Count', 'TOTAL_TAT': 'Avg TAT (hours)'})
            daily_summary.to_excel(writer, sheet_name='Daily Summary')

            # Sheet 3: 검사별 요약
            test_summary = df.groupby('TEST_NAME').agg({
                'SPECIMEN_ID': 'count',
                'TOTAL_TAT': 'mean'
            }).rename(columns={'SPECIMEN_ID': 'Count', 'TOTAL_TAT': 'Avg TAT (hours)'})
            test_summary.to_excel(writer, sheet_name='Test Summary')

        print(f"\n✓ Daily report saved to: {excel_path}")

    except ImportError:
        # openpyxl이 없으면 CSV로 저장
        df.to_csv(output_path, index=False)
        print(f"\n✓ Daily report saved to: {output_path} (CSV format)")
        print("  (Install openpyxl for Excel format: pip install openpyxl)")


def main():
    """메인 실행 함수"""
    print("\nLIMS Daily KPI Dashboard - Running Analysis...")
    print("=" * 70)

    # 1. 샘플 데이터 생성
    lims_data = generate_sample_lims_data()

    # 2. KPI 대시보드 실행
    kpi_metrics = lims_kpi_dashboard(lims_data, show_plots=True)

    # 3. 지연 케이스 탐지
    delayed_cases = detect_delayed_cases(lims_data, tat_threshold=24)

    # 4. 리포트 저장 (Excel 또는 CSV)
    export_daily_report(lims_data, output_path='lims_daily_report.csv')

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  - lims_kpi_dashboard.png      (Dashboard visualization)")
    print("  - lims_daily_report.csv       (Detailed CSV report)")
    print("\nKPI Summary:")
    for key, value in kpi_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    return kpi_metrics


if __name__ == "__main__":
    # 실행
    results = main()
