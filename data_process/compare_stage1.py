import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体，根据系统可能需要调整
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def parse_pg_output(filepath):
    """
    Parses the pg_test1.txt file to extract run details and summary statistics.
    """
    run_data = []
    summary_data = {}

    with open(filepath, 'r', encoding='utf-8') as f:  # PG文件是英文，通常用utf-8
        lines = f.readlines()

    in_data_section = False
    in_summary_section = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "Run ID | Success | Episode Achieved" in line:
            in_data_section = True
            continue
        elif "Statistical Summary" in line:
            in_data_section = False
            in_summary_section = True
            continue
        elif "Report End" in line:
            in_summary_section = False
            break  # Stop parsing after summary

        if in_data_section and line.startswith('---') is False:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 3:  # Ensure it's a data row
                run_id_str = parts[0]
                success_str = parts[1]
                episode_str = parts[2]

                run_id = int(re.search(r'\d+', run_id_str).group()) if re.search(r'\d+', run_id_str) else None
                success = True if success_str.lower() == 'yes' else False
                episode = int(episode_str) if episode_str != '--' else 200  # Failures counted as 200

                if run_id is not None:
                    run_data.append({
                        'Run ID': run_id,
                        'Success': success,
                        'Episode Achieved': episode
                    })
        elif in_summary_section:
            if "Total Runs:" in line:
                summary_data['Total Runs'] = int(line.split(':')[-1].strip())
            elif "Success Count (Reached 500 frames):" in line:
                summary_data['Success Count'] = int(line.split(':')[-1].strip())
            elif "Failure Count:" in line:
                summary_data['Failure Count'] = int(line.split(':')[-1].strip())
            elif "Success Rate:" in line:
                summary_data['Success Rate'] = float(line.split(':')[-1].strip().replace('%', ''))
            elif "Average Episode to Achieve 500 Frames (Overall, failures counted as 200):" in line:
                summary_data['Average Episode (Overall)'] = float(line.split(':')[-1].strip())
            elif "Worst Episode (Overall max):" in line:
                summary_data['Worst Episode (Overall)'] = int(line.split(':')[-1].strip())
            elif "Best Episode (min among successful runs):" in line:
                summary_data['Best Episode (Successful)'] = int(line.split(':')[-1].strip())

    df = pd.DataFrame(run_data)
    return df, summary_data


def parse_dqn_output(filepath):
    """
    Parses the dqn_test1.txt file to extract run details and summary statistics.
    Handles Chinese characters.
    """
    run_data = []
    summary_data = {}

    # 明确指定编码为 'gbk'
    with open(filepath, 'r', encoding='gbk') as f:
        lines = f.readlines()

    in_data_section = False
    in_summary_section = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "运行次数 | 是否成功 | 达成episode" in line:
            in_data_section = True
            continue
        elif "统计摘要" in line:
            in_data_section = False
            in_summary_section = True
            continue
        elif "报告结束" in line:
            in_summary_section = False
            break  # Stop parsing after summary

        if in_data_section and line.startswith('---') is False:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 3:  # Ensure it's a data row
                run_id_str = parts[0]
                success_str = parts[1]
                episode_str = parts[2]

                run_id = int(re.search(r'\d+', run_id_str).group()) if re.search(r'\d+', run_id_str) else None
                success = True if success_str == '成功' else False
                episode = int(episode_str)  # DQN data already numerical for episodes

                if run_id is not None:
                    run_data.append({
                        'Run ID': run_id,
                        'Success': success,
                        'Episode Achieved': episode
                    })
        elif in_summary_section:
            if "总运行次数:" in line:
                summary_data['Total Runs'] = int(line.split(':')[-1].strip())
            elif "成功次数 (达到500帧):" in line:
                summary_data['Success Count'] = int(line.split(':')[-1].strip())
            elif "失败次数:" in line:
                summary_data['Failure Count'] = int(line.split(':')[-1].strip())
            # DQN summary doesn't explicitly state "Success Rate", calculate from counts
            # For average and best, need to be careful as '失败记为200' is mentioned.
            elif "平均达成episode (总体, 失败记为200):" in line:
                summary_data['Average Episode (Overall)'] = float(line.split(':')[-1].strip())
            elif "最差成绩 (总体):" in line:
                summary_data['Worst Episode (Overall)'] = int(line.split(':')[-1].strip())
            elif "最佳成绩 (成功运行):" in line:
                # Need to strip the (episode越小越好) part if present
                val_str = line.split(':')[-1].strip().split('(')[0].strip()
                summary_data['Best Episode (Successful)'] = int(val_str)

    # Calculate Success Rate if not explicitly present in summary
    if 'Total Runs' in summary_data and 'Success Count' in summary_data:
        summary_data['Success Rate'] = (summary_data['Success Count'] / summary_data['Total Runs']) * 100 if \
        summary_data['Total Runs'] > 0 else 0

    df = pd.DataFrame(run_data)
    return df, summary_data


def visualize_comparison(pg_df, pg_summary, dqn_df, dqn_summary, output_dir="plots"):
    """
    Generates and saves comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Success Rate Comparison ---
    success_rates = pd.DataFrame({
        'Algorithm': ['策略梯度', 'DQN'],  # 改为中文
        'Success Rate (%)': [pg_summary.get('Success Rate', 0), dqn_summary.get('Success Rate', 0)]
    })

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Algorithm', y='Success Rate (%)', data=success_rates, palette='viridis')
    plt.ylim(0, 100)
    # 标题修改为中文并添加说明
    plt.title('算法成功率对比 (改进前首次对比)')
    plt.ylabel('成功率 (%)')  # 改为中文
    plt.xlabel('算法')  # 改为中文
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_comparison_第一次对比_原始版本.png'))  # 文件名加_cn
    plt.close()

    # --- 2. Key Performance Metrics Comparison (Average Overall & Best Successful) ---
    metrics_data = pd.DataFrame({
        'Metric': ['平均达成Episode数 (总体)', '最佳达成Episode数 (成功运行)'],  # 改为中文
        '策略梯度': [pg_summary.get('Average Episode (Overall)', 0), pg_summary.get('Best Episode (Successful)', 0)],
        # 改为中文
        'DQN': [dqn_summary.get('Average Episode (Overall)', 0), dqn_summary.get('Best Episode (Successful)', 0)]
    })

    metrics_melted = metrics_data.melt(id_vars='Metric', var_name='Algorithm', value_name='Episode Count')

    plt.figure(figsize=(10, 7))
    sns.barplot(x='Metric', y='Episode Count', hue='Algorithm', data=metrics_melted, palette='plasma')
    # 标题修改为中文并添加说明
    plt.title('算法性能指标对比 (改进前首次对比, Episode 数值越低越好)')
    plt.ylabel('达成Episode数 (越低越好)')  # 改为中文
    plt.xlabel('指标')  # 改为中文
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='算法')  # 图例标题改为中文
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'key_metrics_comparison_第一次对比_原始版本.png'))  # 文件名加_cn
    plt.close()

    # --- 3. Distribution of Successful Episodes (Box Plot) ---
    # Filter for successful runs only and combine
    pg_successful_episodes = pg_df[pg_df['Success'] == True]['Episode Achieved'].tolist()
    dqn_successful_episodes = dqn_df[dqn_df['Success'] == True]['Episode Achieved'].tolist()

    combined_successful_episodes = pd.DataFrame({  # Renamed for clarity
        'Episode Achieved': pg_successful_episodes + dqn_successful_episodes,
        'Algorithm': ['策略梯度'] * len(pg_successful_episodes) + ['DQN'] * len(dqn_successful_episodes)  # 改为中文
    })

    plt.figure(figsize=(10, 7))
    sns.boxplot(x='Algorithm', y='Episode Achieved', data=combined_successful_episodes, palette='coolwarm')
    sns.stripplot(x='Algorithm', y='Episode Achieved', data=combined_successful_episodes, color='black', size=4,
                  jitter=True, alpha=0.6)  # Add individual points
    plt.title('成功运行Episode数分布 (改进前首次对比, Episode 数值越低越好)')
    plt.ylabel('达成Episode数 (越低越好)')  # 改为中文
    plt.xlabel('算法')  # 改为中文
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'successful_episodes_distribution_第一次对比_原始版本.png'))  # 文件名加_cn
    plt.close()

    # --- NEW PLOT 1: Distribution of ALL Episodes (Histogram + KDE) ---
    # Combine all runs data
    pg_all_episodes = pg_df['Episode Achieved'].tolist()
    dqn_all_episodes = dqn_df['Episode Achieved'].tolist()

    combined_all_episodes = pd.DataFrame({
        'Episode Achieved': pg_all_episodes + dqn_all_episodes,
        'Algorithm': ['策略梯度'] * len(pg_all_episodes) + ['DQN'] * len(dqn_all_episodes)
    })

    plt.figure(figsize=(12, 7))
    sns.histplot(data=combined_all_episodes, x='Episode Achieved', hue='Algorithm',
                 kde=True, palette='viridis', bins=10, alpha=0.6, multiple='layer')
    plt.title('所有运行Episode数分布 (改进前首次对比, 失败记为200, 数值越低越好)')
    plt.xlabel('达成Episode数 (越低越好)')
    plt.ylabel('频率')
    plt.legend(title='算法', labels=['DQN', '策略梯度'])  # 确保图例显示正确
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_episodes_distribution_hist_kde_第一次对比_原始版本.png'))
    plt.close()

    # --- NEW PLOT 2: Scatter Plot of All Runs, Differentiating Success/Failure ---
    # Add a 'Status' column to the combined dataframe
    pg_df_temp = pg_df.copy()
    pg_df_temp['Algorithm'] = '策略梯度'
    pg_df_temp['Status'] = pg_df_temp['Success'].apply(lambda x: '成功' if x else '失败')

    dqn_df_temp = dqn_df.copy()
    dqn_df_temp['Algorithm'] = 'DQN'
    dqn_df_temp['Status'] = dqn_df_temp['Success'].apply(lambda x: '成功' if x else '失败')

    all_runs_detailed = pd.concat([pg_df_temp, dqn_df_temp])

    plt.figure(figsize=(12, 7))
    sns.stripplot(x='Algorithm', y='Episode Achieved', hue='Status', data=all_runs_detailed,
                  palette={'成功': 'green', '失败': 'red'}, jitter=True, size=7, alpha=0.8,
                  marker='o', edgecolor='gray', linewidth=0.5)

    # Optionally add a boxplot or violinplot behind for context
    # sns.boxplot(x='Algorithm', y='Episode Achieved', data=all_runs_detailed, palette='light:gray', boxprops=dict(alpha=0.2), showfliers=False)
    # sns.violinplot(x='Algorithm', y='Episode Achieved', data=all_runs_detailed, palette='light:gray', inner=None, alpha=0.2)

    plt.title('所有运行Episode数散点图 (改进前首次对比, 失败记为200, 数值越低越好)')
    plt.xlabel('算法')
    plt.ylabel('达成Episode数 (越低越好)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='运行状态')  # 图例标题
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_runs_scatter_plot_第一次对比_原始版本.png'))
    plt.close()

    print(f"Comparison plots saved to: {output_dir}")


if __name__ == "__main__":
    # Get the directory of the current script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct relative paths to input files
    pg_filepath = os.path.join(current_script_dir, '..', 'pg_output', 'pg_test1.txt')
    dqn_filepath = os.path.join(current_script_dir, '..', 'dqn_output', 'dqn_test1.txt')

    # Output directory for plots
    output_plot_dir = os.path.join(current_script_dir, 'plots')

    print(f"Parsing Policy Gradient data from: {pg_filepath}")
    pg_df, pg_summary = parse_pg_output(pg_filepath)
    print("Policy Gradient Summary:", pg_summary)

    print(f"\nParsing DQN data from: {dqn_filepath}")
    dqn_df, dqn_summary = parse_dqn_output(dqn_filepath)
    print("DQN Summary:", dqn_summary)

    print("\nGenerating visualizations...")
    visualize_comparison(pg_df, pg_summary, dqn_df, dqn_summary, output_plot_dir)
    print("Process finished.")