from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


class MetricExporter:
    """
    Data Analysis utility for publication-ready artifact extraction.

    This class enables researchers to automate the generation of LaTeX tables
    and CSV/JSON artifacts derived from MARL experimental data, specifically
    focusing on Sim2Real retention gap and SLA Uptime.
    """

    def __init__(self, output_dir: str = 'artifacts/publication'):
        """
        Initializes the exporter with a target storage directory.

        Args:
            output_dir (str): Relative path for storing exported tables and results.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f'[MetricExporter] Publication artifacts will be stored in: {self.output_dir}'
        )

    def export_summary_table(
        self,
        experiment_results: List[Dict[str, float]],
        filename: str = 'summary_results',
    ) -> None:
        """
        Converts a list of experiment results into CSV and LaTeX tables.

        Args:
            experiment_results (List[Dict[str, float]]): Each dict represents one
                seed or scenario run with metric:value pairs.
            filename (str): Name for the output files (no extension).
        """
        df = pd.DataFrame(experiment_results)

        # 1. Export as CSV (Raw artifact)
        csv_path = self.output_dir / f'{filename}.csv'
        df.to_csv(csv_path, index=False)

        # 2. Export as LaTeX (Publication-ready)
        # Use mean and standard deviation for the research paper
        summary_stats = df.agg(['mean', 'std', 'median']).T
        latex_path = self.output_dir / f'{filename}_table.tex'

        # Prepare for LaTeX display
        summary_stats.to_latex(
            latex_path,
            index=True,
            column_format='|l|c|c|c|',
            header=['Mean', 'StdDev', 'Median'],
            caption='Comparison of CT-GMARL Performance Across Key Cyber Metrics (10 Seeds)',
            label='tab:marl_performance_summary',
        )

        print(
            f'[MetricExporter] Successfully exported publication artifacts to {self.output_dir}'
        )

    def calculate_retention_gap(
        self, mock_reward: float, docker_reward: float, metric_name: str = 'Reward'
    ) -> Dict[str, float]:
        """
        Computes the Sim2Real Retention Gap for publications.

        Args:
            mock_reward (float): Metric value in the MockHypervisor.
            docker_reward (float): Metric value in the DockerHypervisor.
            metric_name (str): Label for the calculated gap.

        Returns:
            Dict[str, float]: A dictionary with the gap percentage and raw delta.
        """
        delta = mock_reward - docker_reward
        retention_gap = (delta / abs(mock_reward)) * 100

        return {
            f'{metric_name}_retention_gap_pct': retention_gap,
            f'{metric_name}_sim2real_delta': delta,
        }

    def compute_sla_compliance(
        self, online_nodes_history: List[int], total_nodes: int
    ) -> float:
        """
        Calculates the SLA Uptime percentage over an episode.

        Args:
            online_nodes_history (List[int]): Frequency of online nodes at each tick.
            total_nodes (int): Total number of business nodes in the topology.

        Returns:
            float: Percentage of total compliance [0.0 - 100.0].
        """
        if not online_nodes_history or total_nodes == 0:
            return 0.0

        compliance_ratios = [nodes / total_nodes for nodes in online_nodes_history]
        return np.mean(compliance_ratios) * 100
