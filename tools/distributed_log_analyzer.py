import argparse
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import json

class DistributedLogAnalyzer:
    """Analyzer for distributed training logs to help debug issues across ranks."""
    
    def __init__(self, log_dir: str, pattern: str = "rank_*.log"):
        """Initialize the log analyzer.
        
        Args:
            log_dir: Directory containing log files
            pattern: Glob pattern to match log files
        """
        self.log_dir = Path(log_dir)
        self.pattern = pattern
        self.log_files = list(self.log_dir.glob(pattern))
        
        if not self.log_files:
            raise ValueError(f"No log files found in {log_dir} matching pattern {pattern}")
            
        self.log_data = {}  # Will store parsed log data by rank
        self.rank_info = {}  # Will store information about each rank
        
        self._parse_logs()
    
    def _parse_logs(self):
        """Parse all log files and extract structured information."""
        log_pattern = r'\[(.*?)\]\[Rank (\d+)\] (\w+): (.*)'
        
        for log_file in self.log_files:
            rank = None
            entries = []
            
            with open(log_file, 'r') as f:
                for line in f:
                    match = re.match(log_pattern, line.strip())
                    if match:
                        timestamp_str, rank_str, level, message = match.groups()
                        rank = int(rank_str)
                        
                        # Parse timestamp to datetime object
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            # If parsing fails, use the string
                            timestamp = timestamp_str
                            
                        entries.append({
                            'timestamp': timestamp,
                            'level': level,
                            'message': message,
                            'raw_line': line.strip()
                        })
            
            if rank is not None:
                self.log_data[rank] = entries
                self.rank_info[rank] = {
                    'log_file': log_file,
                    'entry_count': len(entries)
                }
    
    def get_rank_summary(self):
        """Get a summary of ranks and their log files."""
        summary = []
        for rank, info in sorted(self.rank_info.items()):
            summary.append({
                'rank': rank,
                'log_file': info['log_file'].name,
                'entries': info['entry_count'],
                'levels': self._count_log_levels(rank)
            })
        return pd.DataFrame(summary)
    
    def _count_log_levels(self, rank):
        """Count occurrences of each log level for a rank."""
        level_counts = {}
        for entry in self.log_data[rank]:
            level = entry['level']
            level_counts[level] = level_counts.get(level, 0) + 1
        return level_counts
    
    def find_timeline_discrepancies(self, min_gap_seconds=5):
        """Find major timeline discrepancies between ranks.
        
        Args:
            min_gap_seconds: Minimum gap in seconds to report as a discrepancy
            
        Returns:
            DataFrame with discrepancy information
        """
        # Extract key events and their timestamps for each rank
        key_events = defaultdict(dict)
        
        for rank, entries in self.log_data.items():
            for entry in entries:
                # Skip non-datetime timestamps
                if not isinstance(entry['timestamp'], datetime):
                    continue
                    
                # Store the timestamp for this message and rank
                message = entry['message']
                key_events[message][rank] = entry['timestamp']
        
        # Find events that have significant timing differences
        discrepancies = []
        
        for event, rank_times in key_events.items():
            if len(rank_times) > 1:  # Only consider events logged by multiple ranks
                timestamps = list(rank_times.values())
                min_time = min(timestamps)
                max_time = max(timestamps)
                
                # Calculate time difference in seconds
                time_diff = (max_time - min_time).total_seconds()
                
                if time_diff >= min_gap_seconds:
                    earliest_rank = [r for r, t in rank_times.items() if t == min_time][0]
                    latest_rank = [r for r, t in rank_times.items() if t == max_time][0]
                    
                    discrepancies.append({
                        'event': event,
                        'time_diff_seconds': time_diff,
                        'earliest_rank': earliest_rank,
                        'latest_rank': latest_rank,
                        'ranks_reporting': len(rank_times),
                        'total_ranks': len(self.log_data)
                    })
        
        return pd.DataFrame(discrepancies).sort_values('time_diff_seconds', ascending=False)
    
    def search_across_ranks(self, pattern):
        """Search for a pattern across all rank logs.
        
        Args:
            pattern: Regex pattern to search for
            
        Returns:
            DataFrame with search results
        """
        results = []
        
        for rank, entries in self.log_data.items():
            for entry in entries:
                if re.search(pattern, entry['message'], re.IGNORECASE) or \
                   re.search(pattern, entry['raw_line'], re.IGNORECASE):
                    results.append({
                        'rank': rank,
                        'timestamp': entry['timestamp'],
                        'level': entry['level'],
                        'message': entry['message'],
                        'raw_line': entry['raw_line']
                    })
        
        return pd.DataFrame(results)
    
    def find_missing_messages(self, pattern=None):
        """Find messages that are logged by some ranks but missing in others.
        
        Args:
            pattern: Optional regex to filter messages
            
        Returns:
            Dictionary mapping messages to ranks that logged them
        """
        # Extract messages from each rank
        rank_messages = defaultdict(set)
        
        for rank, entries in self.log_data.items():
            for entry in entries:
                message = entry['message']
                
                # Apply filter if provided
                if pattern and not re.search(pattern, message, re.IGNORECASE):
                    continue
                    
                rank_messages[rank].add(message)
        
        # Find all unique messages
        all_messages = set()
        for messages in rank_messages.values():
            all_messages.update(messages)
        
        # Find which ranks logged each message
        message_to_ranks = {msg: [] for msg in all_messages}
        for rank, messages in rank_messages.items():
            for msg in messages:
                message_to_ranks[msg].append(rank)
        
        # Filter to only include messages not logged by all ranks
        inconsistent_messages = {}
        for msg, ranks in message_to_ranks.items():
            if len(ranks) < len(self.log_data):
                inconsistent_messages[msg] = {
                    'ranks_with_message': sorted(ranks),
                    'ranks_missing_message': sorted(set(self.log_data.keys()) - set(ranks))
                }
        
        return inconsistent_messages
    
    def compare_initialization_data(self):
        """Compare initialization data between ranks to find inconsistencies.
        
        Returns:
            Dictionary with initialization parameters and their values by rank
        """
        # Common initialization parameters to look for
        init_patterns = [
            r'Initialized dataset with (\d+) samples',
            r'Training on (\d+) nodes, (\d+) devices per node',
            r'Using (\w+) strategy with (\w+) precision',
            r'Training on GPU: (.*)',
            r'CUDA_VISIBLE_DEVICES: (.*)',
            r'Starting epoch (\d+)/(\d+)'
        ]
        
        # Extract initialization data from each rank
        init_data = defaultdict(dict)
        
        for rank, entries in self.log_data.items():
            for entry in entries:
                message = entry['message']
                
                for pattern in init_patterns:
                    match = re.search(pattern, message)
                    if match:
                        # Use the pattern as the key and the captured groups as the value
                        key = re.sub(r'\(.*?\)', '{}', pattern)
                        values = match.groups()
                        init_data[key][rank] = values
        
        # Find inconsistencies
        inconsistencies = {}
        
        for param, rank_values in init_data.items():
            # Group ranks by their values
            value_to_ranks = defaultdict(list)
            for rank, value in rank_values.items():
                value_tuple = tuple(value)  # Make value hashable
                value_to_ranks[value_tuple].append(rank)
            
            # If there's more than one unique value, we have an inconsistency
            if len(value_to_ranks) > 1:
                inconsistencies[param] = {
                    str(value): sorted(ranks) for value, ranks in value_to_ranks.items()
                }
        
        return {
            'all_initialization_data': init_data,
            'inconsistencies': inconsistencies
        }
    
    def find_errors_and_warnings(self):
        """Extract all errors and warnings from logs.
        
        Returns:
            DataFrame with error and warning information
        """
        error_levels = ['ERROR', 'CRITICAL', 'FATAL', 'WARNING', 'WARN']
        issues = []
        
        for rank, entries in self.log_data.items():
            for entry in entries:
                if entry['level'] in error_levels:
                    issues.append({
                        'rank': rank,
                        'timestamp': entry['timestamp'],
                        'level': entry['level'],
                        'message': entry['message']
                    })
        
        return pd.DataFrame(issues)
    
    def analyze_training_progress(self):
        """Analyze training progress across ranks.
        
        Looks for epoch start/end messages and extracts timing information.
        
        Returns:
            DataFrame with training progress information
        """
        # Patterns to identify epoch start and end
        epoch_start_pattern = r'Starting epoch (\d+)/(\d+)'
        epoch_end_pattern = r'Finished epoch (\d+)/(\d+)'
        
        progress_data = []
        
        for rank, entries in self.log_data.items():
            epoch_starts = {}
            
            for entry in entries:
                message = entry['message']
                timestamp = entry['timestamp']
                
                # Skip non-datetime timestamps
                if not isinstance(timestamp, datetime):
                    continue
                
                # Check for epoch start
                start_match = re.search(epoch_start_pattern, message)
                if start_match:
                    epoch = int(start_match.group(1))
                    epoch_starts[epoch] = timestamp
                
                # Check for epoch end
                end_match = re.search(epoch_end_pattern, message)
                if end_match:
                    epoch = int(end_match.group(1))
                    if epoch in epoch_starts:
                        duration = (timestamp - epoch_starts[epoch]).total_seconds()
                        
                        progress_data.append({
                            'rank': rank,
                            'epoch': epoch,
                            'start_time': epoch_starts[epoch],
                            'end_time': timestamp,
                            'duration_seconds': duration
                        })
        
        return pd.DataFrame(progress_data)
    
    def plot_training_durations(self):
        """Plot training durations per epoch across ranks.
        
        Returns:
            Matplotlib figure with the plot
        """
        progress_df = self.analyze_training_progress()
        
        if progress_df.empty:
            print("No training progress data found in logs")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ranks = sorted(progress_df['rank'].unique())
        for rank in ranks:
            rank_data = progress_df[progress_df['rank'] == rank]
            ax.plot(rank_data['epoch'], rank_data['duration_seconds'], 
                    marker='o', label=f'Rank {rank}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Duration (seconds)')
        ax.set_title('Training Duration per Epoch Across Ranks')
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def generate_report(self, output_file=None):
        """Generate a comprehensive report of log analysis.
        
        Args:
            output_file: Path to save the report (HTML format)
            
        Returns:
            HTML string with the report
        """
        # Create a report using pandas styling
        sections = []
        
        # 1. Basic summary
        sections.append("<h2>Rank Summary</h2>")
        sections.append(self.get_rank_summary().to_html())
        
        # 2. Timeline discrepancies
        sections.append("<h2>Timeline Discrepancies</h2>")
        discrepancies = self.find_timeline_discrepancies()
        if not discrepancies.empty:
            sections.append(discrepancies.to_html())
        else:
            sections.append("<p>No significant timeline discrepancies found.</p>")
        
        # 3. Initialization data inconsistencies
        sections.append("<h2>Initialization Inconsistencies</h2>")
        init_data = self.compare_initialization_data()
        if init_data['inconsistencies']:
            sections.append("<pre>" + json.dumps(init_data['inconsistencies'], indent=2) + "</pre>")
        else:
            sections.append("<p>No initialization inconsistencies found.</p>")
        
        # 4. Errors and warnings
        sections.append("<h2>Errors and Warnings</h2>")
        issues = self.find_errors_and_warnings()
        if not issues.empty:
            sections.append(issues.to_html())
        else:
            sections.append("<p>No errors or warnings found.</p>")
        
        # 5. Training progress
        sections.append("<h2>Training Progress</h2>")
        progress = self.analyze_training_progress()
        if not progress.empty:
            sections.append(progress.to_html())
        else:
            sections.append("<p>No training progress data found.</p>")
        
        # Combine all sections
        report = "<html><body>" + "".join(sections) + "</body></html>"
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze distributed training logs')
    parser.add_argument('log_dir', help='Directory containing rank log files')
    parser.add_argument('--pattern', default='rank_*.log', help='Log file pattern to match')
    parser.add_argument('--report', help='Output file for HTML report')
    parser.add_argument('--search', help='Search pattern across all logs')
    parser.add_argument('--min-gap', type=int, default=5, 
                        help='Minimum time gap (seconds) to report as discrepancy')
    parser.add_argument('--plot', action='store_true', 
                        help='Generate and display plots')
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        analyzer = DistributedLogAnalyzer(args.log_dir, args.pattern)
        
        print(f"Found {len(analyzer.log_files)} log files")
        print("\nRank Summary:")
        print(analyzer.get_rank_summary())
        
        if args.search:
            print(f"\nSearch results for '{args.search}':")
            print(analyzer.search_across_ranks(args.search))
        
        print("\nTimeline Discrepancies:")
        print(analyzer.find_timeline_discrepancies(args.min_gap))
        
        print("\nInitialization Inconsistencies:")
        init_data = analyzer.compare_initialization_data()
        if init_data['inconsistencies']:
            print(json.dumps(init_data['inconsistencies'], indent=2))
        else:
            print("No initialization inconsistencies found")
        
        print("\nErrors and Warnings:")
        issues = analyzer.find_errors_and_warnings()
        if not issues.empty:
            print(issues)
        else:
            print("No errors or warnings found")
        
        if args.plot:
            fig = analyzer.plot_training_durations()
            if fig:
                plt.show()
        
        if args.report:
            analyzer.generate_report(args.report)
            print(f"\nReport generated: {args.report}")
    
    except Exception as e:
        print(f"Error analyzing logs: {e}")
        raise

if __name__ == "__main__":
    main() 