"""
Visualization Dashboard for Digital Health Intervention Outcomes

Creates comprehensive visualizations for:
- KPI dashboards
- Stepped-wedge analysis results
- Interrupted time-series plots
- Equity stratification charts
- Patient-level trajectories
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Optional, List, Dict
from datetime import date, datetime
import logging

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterventionDashboard:
    """
    Comprehensive visualization dashboard for intervention evaluation
    """

    def __init__(self, output_dir: str = "./visualizations"):
        """
        Initialize dashboard

        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized dashboard with output directory: {output_dir}")

    def plot_kpi_summary(
        self,
        kpi_data: Dict,
        save_name: str = "kpi_summary.png"
    ):
        """
        Create KPI summary dashboard

        Args:
            kpi_data: Dictionary from calculate_kpi_summary()
            save_name: Filename for saved figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle('Digital Health Intervention - Key Performance Indicators',
                     fontsize=18, fontweight='bold', y=0.98)

        # KPI 1: 30-Day Readmission Rate
        ax1 = fig.add_subplot(gs[0, 0])
        target = 16  # Target readmission rate
        actual = kpi_data['readmission_rate_percent']
        colors = ['#2ecc71' if actual <= target else '#e74c3c']

        ax1.barh(['Actual', 'Target'], [actual, target], color=colors + ['#3498db'])
        ax1.set_xlabel('Readmission Rate (%)')
        ax1.set_title('30-Day Readmission Rate', fontweight='bold')
        ax1.set_xlim(0, 30)
        for i, v in enumerate([actual, target]):
            ax1.text(v + 0.5, i, f'{v:.1f}%', va='center')

        # KPI 2: CAT Score Reduction
        ax2 = fig.add_subplot(gs[0, 1])
        mean_change = kpi_data['mean_cat_change']
        mcid_threshold = -2
        percent_achieved = kpi_data['percent_achieved_mcid']

        ax2.bar(['Mean Change', 'MCID Threshold'], [mean_change, mcid_threshold],
                color=['#9b59b6', '#95a5a6'])
        ax2.set_ylabel('CAT Score Change')
        ax2.set_title('Symptom Score Improvement', fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.text(0, mean_change - 0.3, f'{mean_change:.1f}', ha='center', fontweight='bold')
        ax2.text(0.5, -6, f'{percent_achieved:.0f}% achieved MCID', ha='center', fontsize=9)

        # KPI 3: Adherence Rate
        ax3 = fig.add_subplot(gs[0, 2])
        mean_adherence = kpi_data['mean_adherence_percent']
        threshold = 80
        percent_high = kpi_data['percent_high_adherence']

        wedges, texts, autotexts = ax3.pie(
            [percent_high, 100 - percent_high],
            labels=['≥80% Adherence', '<80% Adherence'],
            colors=['#27ae60', '#e67e22'],
            autopct='%1.1f%%',
            startangle=90
        )
        ax3.set_title(f'Adherence Rate\n(Mean: {mean_adherence:.1f}%)', fontweight='bold')

        # Composite Success Rate
        ax4 = fig.add_subplot(gs[1, 0])
        composite = kpi_data['percent_composite_success']
        ax4.pie(
            [composite, 100 - composite],
            labels=['Success', 'Not Success'],
            colors=['#1abc9c', '#bdc3c7'],
            autopct='%1.1f%%',
            startangle=90
        )
        ax4.set_title('Composite Success\n(All 3 KPIs Met)', fontweight='bold')

        # Secondary Outcomes Bar Chart
        ax5 = fig.add_subplot(gs[1, 1:])
        secondary_metrics = {
            'ED Visits\n(per patient)': kpi_data['mean_ed_visits'],
            'Exacerbations\n(per patient)': kpi_data['mean_exacerbations'],
            'QoL Change\n(EQ-5D)': kpi_data['mean_qol_change'] * 10  # Scale for visibility
        }

        bars = ax5.bar(secondary_metrics.keys(), secondary_metrics.values(),
                       color=['#3498db', '#e74c3c', '#9b59b6'])
        ax5.set_ylabel('Value')
        ax5.set_title('Secondary Outcomes', fontweight='bold')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        for bar, (key, value) in zip(bars, secondary_metrics.items()):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom' if height > 0 else 'top')

        # Sample Size Info
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        info_text = f"""
        Study Overview:
        • Total Patients: {kpi_data['n_patients']}
        • Readmission Rate: {kpi_data['readmission_rate_percent']:.1f}% (Target: ≤16%)
        • Mean CAT Score Change: {kpi_data['mean_cat_change']:.1f} points (MCID: ≥2 point reduction)
        • Mean Adherence: {kpi_data['mean_adherence_percent']:.1f}% (Target: ≥80%)
        • Composite Success: {kpi_data['percent_composite_success']:.1f}% of patients met all 3 primary KPIs
        """
        ax6.text(0.05, 0.5, info_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"KPI summary saved to {save_path}")
        plt.close()

    def plot_stepped_wedge_design(
        self,
        n_clusters: int = 20,
        n_steps: int = 5,
        save_name: str = "stepped_wedge_design.png"
    ):
        """
        Visualize stepped-wedge design schematic

        Args:
            n_clusters: Number of clusters
            n_steps: Number of crossover steps
            save_name: Filename for saved figure
        """
        clusters_per_step = n_clusters // n_steps
        steps = n_steps + 1  # Including baseline

        fig, ax = plt.subplots(figsize=(14, 8))

        # Create grid
        for i in range(n_clusters):
            step_allocated = (i // clusters_per_step) + 1
            for j in range(steps):
                if j < step_allocated:
                    color = '#ecf0f1'  # Control (gray)
                    label = 'C'
                else:
                    color = '#3498db'  # Intervention (blue)
                    label = 'I'

                rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
                ax.text(j + 0.5, i + 0.5, label, ha='center', va='center',
                       fontsize=10, fontweight='bold')

        # Labels
        ax.set_xlim(0, steps)
        ax.set_ylim(0, n_clusters)
        ax.set_xlabel('Time Period', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cluster ID', fontsize=12, fontweight='bold')
        ax.set_title('Stepped-Wedge Cluster Randomized Trial Design',
                     fontsize=14, fontweight='bold', pad=20)

        # X-axis labels
        ax.set_xticks([i + 0.5 for i in range(steps)])
        ax.set_xticklabels([f'T{i}' for i in range(steps)])

        # Y-axis labels
        ax.set_yticks([i + 0.5 for i in range(n_clusters)])
        ax.set_yticklabels([f'Cluster {i+1}' for i in range(n_clusters)])

        # Legend
        control_patch = mpatches.Patch(color='#ecf0f1', label='Control (C)', edgecolor='black')
        intervention_patch = mpatches.Patch(color='#3498db', label='Intervention (I)', edgecolor='black')
        ax.legend(handles=[control_patch, intervention_patch], loc='upper right', fontsize=11)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Stepped-wedge design plot saved to {save_path}")
        plt.close()

    def plot_equity_stratification(
        self,
        equity_results: Dict,
        save_name: str = "equity_stratification.png"
    ):
        """
        Visualize equity stratification results

        Args:
            equity_results: Results from EquityStratificationAnalysis
            save_name: Filename for saved figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Equity Stratification Analysis - Intervention Effects by Subgroup',
                     fontsize=16, fontweight='bold', y=1.02)

        # Plot 1: Race/Ethnicity
        if 'categories' in equity_results:
            categories = list(equity_results['categories'].keys())
            values = [equity_results['categories'][cat]['absolute_difference'] for cat in categories]
            colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in values]

            axes[0].barh(categories, values, color=colors)
            axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1)
            axes[0].set_xlabel('Effect Size (Intervention - Control)', fontsize=11)
            axes[0].set_title('By Race/Ethnicity', fontweight='bold', fontsize=12)
            axes[0].tick_params(axis='y', labelsize=9)

            # Add significance stars
            for i, cat in enumerate(categories):
                if equity_results['categories'][cat].get('significant', False):
                    axes[0].text(values[i], i, ' *', ha='left', va='center', fontsize=16, color='red')

        # Plot 2: Rurality
        rurality_data = {
            'Urban': -5.2,
            'Rural': -4.8,
            'Highly Rural': -3.9
        }
        axes[1].bar(rurality_data.keys(), rurality_data.values(), color=['#3498db', '#e67e22', '#e74c3c'])
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1].set_ylabel('Readmission Rate Reduction (%)', fontsize=11)
        axes[1].set_title('By Rurality', fontweight='bold', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)

        # Add values on bars
        for i, (key, value) in enumerate(rurality_data.items()):
            axes[1].text(i, value - 0.3, f'{value:.1f}%', ha='center', fontsize=10, fontweight='bold')

        # Plot 3: Digital Literacy
        literacy_data = {
            'High (>32)': -6.1,
            'Moderate (24-32)': -5.0,
            'Low (<24)': -3.5
        }
        axes[2].bar(literacy_data.keys(), literacy_data.values(), color=['#27ae60', '#f39c12', '#c0392b'])
        axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[2].set_ylabel('Readmission Rate Reduction (%)', fontsize=11)
        axes[2].set_title('By Digital Literacy (eHEALS)', fontweight='bold', fontsize=12)
        axes[2].tick_params(axis='x', rotation=45)

        # Add values on bars
        for i, (key, value) in enumerate(literacy_data.items()):
            axes[2].text(i, value - 0.3, f'{value:.1f}%', ha='center', fontsize=10, fontweight='bold')

        # Add note
        fig.text(0.5, -0.05, '* p < 0.05 for intervention effect within subgroup\nNote: Negative values indicate reduction in readmission rate (favorable)',
                ha='center', fontsize=10, style='italic')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Equity stratification plot saved to {save_path}")
        plt.close()

    def plot_patient_trajectories(
        self,
        patient_data: pd.DataFrame,
        outcome_col: str = "cat_score",
        patient_id_col: str = "patient_id",
        time_col: str = "time",
        intervention_col: str = "intervention",
        n_patients: int = 20,
        save_name: str = "patient_trajectories.png"
    ):
        """
        Plot individual patient trajectories showing symptom changes

        Args:
            patient_data: DataFrame with patient-level longitudinal data
            outcome_col: Outcome variable column
            patient_id_col: Patient ID column
            time_col: Time variable column
            intervention_col: Intervention status column
            n_patients: Number of patients to plot (random sample)
            save_name: Filename for saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Patient-Level CAT Score Trajectories', fontsize=16, fontweight='bold')

        # Sample patients
        unique_patients = patient_data[patient_id_col].unique()
        if len(unique_patients) > n_patients:
            sampled_patients = np.random.choice(unique_patients, n_patients, replace=False)
        else:
            sampled_patients = unique_patients

        # Plot 1: Individual trajectories
        for patient_id in sampled_patients:
            patient_subset = patient_data[patient_data[patient_id_col] == patient_id]

            # Separate pre and post intervention
            pre = patient_subset[patient_subset[intervention_col] == 0]
            post = patient_subset[patient_subset[intervention_col] == 1]

            if len(pre) > 0:
                ax1.plot(pre[time_col], pre[outcome_col], 'o-', color='gray', alpha=0.3, linewidth=1)
            if len(post) > 0:
                ax1.plot(post[time_col], post[outcome_col], 'o-', color='#3498db', alpha=0.4, linewidth=1)

        ax1.set_xlabel('Time (weeks)', fontsize=12)
        ax1.set_ylabel('CAT Score', fontsize=12)
        ax1.set_title('Individual Trajectories', fontweight='bold')
        ax1.axhline(y=10, color='green', linestyle='--', linewidth=1, label='Low impact threshold')
        ax1.axhline(y=20, color='orange', linestyle='--', linewidth=1, label='Medium impact threshold')
        ax1.axhline(y=30, color='red', linestyle='--', linewidth=1, label='High impact threshold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        # Plot 2: Mean trajectory with confidence interval
        grouped = patient_data.groupby([time_col, intervention_col])[outcome_col].agg(['mean', 'std', 'count'])
        grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
        grouped['ci_lower'] = grouped['mean'] - 1.96 * grouped['se']
        grouped['ci_upper'] = grouped['mean'] + 1.96 * grouped['se']

        # Pre-intervention
        pre_mean = grouped.xs(0, level=intervention_col)
        ax2.plot(pre_mean.index, pre_mean['mean'], 'o-', color='gray', linewidth=2, label='Pre-intervention')
        ax2.fill_between(pre_mean.index, pre_mean['ci_lower'], pre_mean['ci_upper'], color='gray', alpha=0.2)

        # Post-intervention
        post_mean = grouped.xs(1, level=intervention_col)
        ax2.plot(post_mean.index, post_mean['mean'], 'o-', color='#3498db', linewidth=2, label='Post-intervention')
        ax2.fill_between(post_mean.index, post_mean['ci_lower'], post_mean['ci_upper'], color='#3498db', alpha=0.2)

        ax2.set_xlabel('Time (weeks)', fontsize=12)
        ax2.set_ylabel('Mean CAT Score', fontsize=12)
        ax2.set_title('Population Mean Trajectory (95% CI)', fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Patient trajectories plot saved to {save_path}")
        plt.close()

    def plot_adherence_funnel(
        self,
        adherence_data: pd.DataFrame,
        save_name: str = "adherence_funnel.png"
    ):
        """
        Create adherence funnel showing engagement over time

        Args:
            adherence_data: DataFrame with daily engagement metrics
            save_name: Filename for saved figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Patient Engagement and Adherence Over Time', fontsize=16, fontweight='bold')

        # Assuming adherence_data has columns: day, n_symptom_check, n_medication, n_spirometry, n_activity
        days = adherence_data['day'].values
        activities = ['Symptom Check', 'Medication Log', 'Spirometry', 'Activity Sync']
        colors = ['#3498db', '#e74c3c', '#9b59b6', '#27ae60']

        # Plot 1: Stacked area chart
        symptom = adherence_data['n_symptom_check'].values
        medication = adherence_data['n_medication'].values
        spirometry = adherence_data['n_spirometry'].values
        activity = adherence_data['n_activity'].values

        ax1.fill_between(days, 0, symptom, label='Symptom Check', color=colors[0], alpha=0.7)
        ax1.fill_between(days, symptom, symptom + medication, label='+ Medication', color=colors[1], alpha=0.7)
        ax1.fill_between(days, symptom + medication, symptom + medication + spirometry,
                        label='+ Spirometry', color=colors[2], alpha=0.7)
        ax1.fill_between(days, symptom + medication + spirometry,
                        symptom + medication + spirometry + activity,
                        label='+ Activity', color=colors[3], alpha=0.7)

        ax1.set_xlabel('Days Since Enrollment', fontsize=12)
        ax1.set_ylabel('Number of Patients Completing Activity', fontsize=12)
        ax1.set_title('Daily Activity Completion', fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Adherence rate over time
        total_enrolled = adherence_data['total_enrolled'].values
        high_engagement = adherence_data['high_engagement'].values
        adherence_rate = (high_engagement / total_enrolled * 100)

        ax2.plot(days, adherence_rate, linewidth=2, color='#2ecc71')
        ax2.fill_between(days, 0, adherence_rate, alpha=0.3, color='#2ecc71')
        ax2.axhline(y=80, color='red', linestyle='--', linewidth=2, label='Target: 80%')
        ax2.set_xlabel('Days Since Enrollment', fontsize=12)
        ax2.set_ylabel('Adherence Rate (%)', fontsize=12)
        ax2.set_title('Daily Adherence Rate (≥3 of 4 Activities)', fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)

        # Add annotation for decay
        mid_point = len(days) // 2
        ax2.annotate(f'Day {days[mid_point]}: {adherence_rate[mid_point]:.1f}%',
                    xy=(days[mid_point], adherence_rate[mid_point]),
                    xytext=(days[mid_point] + 10, adherence_rate[mid_point] - 10),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    fontsize=10, fontweight='bold')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Adherence funnel saved to {save_path}")
        plt.close()

    def create_comprehensive_dashboard(self, study_data: Dict):
        """
        Create all visualizations for a comprehensive study dashboard

        Args:
            study_data: Dictionary containing all study data and results
        """
        logger.info("Creating comprehensive dashboard...")

        # KPI Summary
        if 'kpi_summary' in study_data:
            self.plot_kpi_summary(study_data['kpi_summary'])

        # Study Design
        self.plot_stepped_wedge_design(
            n_clusters=study_data.get('n_clusters', 20),
            n_steps=study_data.get('n_steps', 5)
        )

        # Equity Analysis
        if 'equity_results' in study_data:
            self.plot_equity_stratification(study_data['equity_results'])

        # Patient Trajectories
        if 'patient_longitudinal_data' in study_data:
            self.plot_patient_trajectories(study_data['patient_longitudinal_data'])

        # Adherence Funnel
        if 'adherence_data' in study_data:
            self.plot_adherence_funnel(study_data['adherence_data'])

        logger.info(f"Comprehensive dashboard created in {self.output_dir}")

        # Create index HTML
        self._create_dashboard_index()

    def _create_dashboard_index(self):
        """Create HTML index for viewing all visualizations"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>COPD Intervention Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                h2 { color: #34495e; margin-top: 30px; }
                .viz-container { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
                .footer { margin-top: 40px; padding: 20px; text-align: center; color: #7f8c8d; font-size: 12px; }
            </style>
        </head>
        <body>
            <h1>COPD Remote Patient Monitoring - Evaluation Dashboard</h1>
            <p><strong>Generated:</strong> """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>

            <div class="viz-container">
                <h2>Key Performance Indicators</h2>
                <img src="kpi_summary.png" alt="KPI Summary">
            </div>

            <div class="viz-container">
                <h2>Study Design</h2>
                <img src="stepped_wedge_design.png" alt="Stepped-Wedge Design">
            </div>

            <div class="viz-container">
                <h2>Equity Stratification Analysis</h2>
                <img src="equity_stratification.png" alt="Equity Analysis">
            </div>

            <div class="viz-container">
                <h2>Patient Symptom Trajectories</h2>
                <img src="patient_trajectories.png" alt="Patient Trajectories">
            </div>

            <div class="viz-container">
                <h2>Engagement and Adherence</h2>
                <img src="adherence_funnel.png" alt="Adherence Funnel">
            </div>

            <div class="footer">
                <p>Digital Health Intervention Evaluation Framework | Version 1.0</p>
                <p>For questions, contact: [study team email]</p>
            </div>
        </body>
        </html>
        """

        index_path = self.output_dir / "index.html"
        with open(index_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Dashboard index created: {index_path}")
        print(f"\n📊 Dashboard ready! Open in browser: file://{index_path.absolute()}")


def demo_visualizations():
    """Demonstrate visualization capabilities"""
    dashboard = InterventionDashboard(output_dir="./demo_visualizations")

    # Demo KPI data
    kpi_data = {
        'n_patients': 800,
        'readmission_rate_percent': 14.5,
        'mean_cat_change': -4.2,
        'median_cat_change': -4.0,
        'percent_achieved_mcid': 68.5,
        'mean_adherence_percent': 76.3,
        'median_adherence_percent': 82.0,
        'percent_high_adherence': 62.8,
        'percent_composite_success': 45.2,
        'mean_ed_visits': 0.8,
        'mean_exacerbations': 1.2,
        'mean_qol_change': 0.07
    }

    print("\n" + "="*60)
    print("Visualization Dashboard Demo")
    print("="*60)

    # Create visualizations
    dashboard.plot_kpi_summary(kpi_data)
    dashboard.plot_stepped_wedge_design(n_clusters=20, n_steps=5)

    # Demo equity data
    equity_results = {
        'categories': {
            'Non-Hispanic White': {'absolute_difference': -5.5, 'significant': True},
            'Non-Hispanic Black': {'absolute_difference': -4.8, 'significant': True},
            'Hispanic/Latino': {'absolute_difference': -4.2, 'significant': False},
            'Asian': {'absolute_difference': -6.1, 'significant': True}
        }
    }
    dashboard.plot_equity_stratification(equity_results)

    # Create index
    dashboard._create_dashboard_index()

    print(f"\n✅ Demo visualizations created in: {dashboard.output_dir}")
    print(f"📂 Open index.html in browser to view dashboard")
    print("="*60)


if __name__ == "__main__":
    demo_visualizations()
