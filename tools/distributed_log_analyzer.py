import argparse
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import json
from typing import Dict, List, Optional, Union, Any, DefaultDict, Tuple

class DistributedLogAnalyzer:
    """Analyzer for distributed training logs to help debug issues across ranks."""
    
    def __init__(self, log_dir: str, pattern: str = "rank_*.log") -> None:
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
            
        self.log_data: Dict[int, List[Dict[str, Any]]] = {}  # Will store parsed log data by rank
        self.rank_info: Dict[int, Dict[str, Any]] = {}  # Will store information about each rank
        
        self._parse_logs()
    
    def _parse_logs(self) -> None:
        """Parse all log files and extract structured information."""
        log_pattern = r'\[(.*?)\]\[Rank (\d+)\] (\w+): (.*)'
        
        for log_file in self.log_files:
            rank = None
            entries: List[Dict[str, Any]] = []
            
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
    
    def get_rank_summary(self) -> pd.DataFrame:
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
    
    def _count_log_levels(self, rank: int) -> Dict[str, int]:
        """Count occurrences of each log level for a rank."""
        level_counts: Dict[str, int] = {}
        for entry in self.log_data[rank]:
            level = entry['level']
            level_counts[level] = level_counts.get(level, 0) + 1
        return level_counts
    
    def _group_events_by_content(self) -> DefaultDict[str, List[int]]:
        """Group log entries by their message content across all ranks.
        
        Returns:
            Dictionary mapping message content to list of ranks that logged it
        """
        events: DefaultDict[str, List[int]] = defaultdict(list)
        for rank, entries in self.log_data.items():
            for entry in entries:
                message = entry['message']
                events[message].append(rank)
        return events
    
    def find_timeline_discrepancies(self, min_gap_seconds: int = 5) -> Union[pd.DataFrame, str]:
        """Find significant time differences between ranks for the same events.
        
        Args:
            min_gap_seconds: Minimum time difference to report as a discrepancy
            
        Returns:
            DataFrame of discrepancies sorted by time difference or message if none found
        """
        discrepancies: List[Dict[str, Any]] = []
        
        # Group events by their message content
        for event, ranks in self._group_events_by_content().items():
            # Get the first timestamp for each rank that logged this event
            rank_times: Dict[int, datetime] = {}
            for rank in ranks:
                # Find the first entry with this message in the rank's logs
                for entry in self.log_data[rank]:
                    if entry['message'] == event:
                        rank_times[rank] = entry['timestamp']
                        break
            
            if len(rank_times) > 1:  # Only check if multiple ranks report the event
                min_time = min(rank_times.values())
                max_time = max(rank_times.values())
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
        
        if not discrepancies:
            return "No timeline discrepancies found"
            
        df = pd.DataFrame(discrepancies)
        return df.sort_values('time_diff_seconds', ascending=False)
    
    def search_across_ranks(self, pattern: str) -> pd.DataFrame:
        """Search for a pattern across all rank logs.
        
        Args:
            pattern: Regex pattern to search for
            
        Returns:
            DataFrame with search results
        """
        results: List[Dict[str, Any]] = []
        
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
    
    def find_missing_messages(self, pattern: Optional[str] = None) -> Dict[str, Dict[str, List[int]]]:
        """Find messages that are logged by some ranks but missing in others.
        
        Args:
            pattern: Optional regex to filter messages
            
        Returns:
            Dictionary mapping messages to ranks that logged them
        """
        # Extract messages from each rank
        rank_messages: DefaultDict[int, set] = defaultdict(set)
        
        for rank, entries in self.log_data.items():
            for entry in entries:
                message = entry['message']
                
                # Apply filter if provided
                if pattern and not re.search(pattern, message, re.IGNORECASE):
                    continue
                    
                rank_messages[rank].add(message)
        
        # Find all unique messages
        all_messages: set = set()
        for messages in rank_messages.values():
            all_messages.update(messages)
        
        # Find which ranks logged each message
        message_to_ranks: Dict[str, List[int]] = {msg: [] for msg in all_messages}
        for rank, messages in rank_messages.items():
            for msg in messages:
                message_to_ranks[msg].append(rank)
        
        # Filter to only include messages not logged by all ranks
        inconsistent_messages: Dict[str, Dict[str, List[int]]] = {}
        for msg, ranks in message_to_ranks.items():
            if len(ranks) < len(self.log_data):
                inconsistent_messages[msg] = {
                    'ranks_with_message': sorted(ranks),
                    'ranks_missing_message': sorted(set(self.log_data.keys()) - set(ranks))
                }
        
        return inconsistent_messages
    
    def compare_initialization_data(self) -> Dict[str, Dict[str, Any]]:
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
        init_data: DefaultDict[str, Dict[int, Tuple[Any, ...]]] = defaultdict(dict)
        
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
        inconsistencies: Dict[str, Dict[str, List[int]]] = {}
        
        for param, rank_values in init_data.items():
            # Group ranks by their values
            value_to_ranks: DefaultDict[Tuple[Any, ...], List[int]] = defaultdict(list)
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
    
    def find_errors_and_warnings(self) -> pd.DataFrame:
        """Extract all errors and warnings from logs.
        
        Returns:
            DataFrame with error and warning information
        """
        error_levels = ['ERROR', 'CRITICAL', 'FATAL', 'WARNING', 'WARN']
        issues: List[Dict[str, Any]] = []
        
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
    
    def analyze_last_lines(self) -> Dict[str, Any]:
        """Analyze the last line of each log file to find ranks with different messages.
        
        Returns:
            Dictionary containing:
            - majority_message: The most common last message
            - minority_messages: Dictionary mapping non-majority messages to their ranks
            - rank_status: Dictionary mapping ranks to their status (✓ or ✗)
        """
        # Get the last message from each rank
        last_messages: Dict[int, str] = {}
        for rank, entries in self.log_data.items():
            if entries:
                last_messages[rank] = entries[-1]['message']
            else:
                last_messages[rank] = "No log entries found"
        
        # Find the most common message
        message_counts: Dict[str, int] = {}
        for message in last_messages.values():
            message_counts[message] = message_counts.get(message, 0) + 1
        
        if not message_counts:
            return {
                'majority_message': None,
                'minority_messages': {},
                'rank_status': {}
            }
        
        majority_message = max(message_counts.items(), key=lambda x: x[1])[0]
        
        # Find all unique messages that aren't part of the majority
        minority_messages: Dict[str, List[int]] = {}
        for rank, message in last_messages.items():
            if message != majority_message:
                if message not in minority_messages:
                    minority_messages[message] = []
                minority_messages[message].append(rank)
        
        # Create rank status dictionary
        rank_status: Dict[int, str] = {
            rank: '✓' if message == majority_message else '✗'
            for rank, message in last_messages.items()
        }
        
        return {
            'majority_message': majority_message,
            'minority_messages': minority_messages,
            'rank_status': rank_status
        }
    
    def analyze_training_progress(self) -> pd.DataFrame:
        """Analyze training progress across ranks.
        
        Looks for epoch start/end messages and extracts timing information.
        
        Returns:
            DataFrame with training progress information
        """
        # Patterns to identify epoch start and end
        epoch_start_pattern = r'Starting epoch (\d+)/(\d+)'
        epoch_end_pattern = r'Finished epoch (\d+)/(\d+)'
        
        progress_data: List[Dict[str, Any]] = []
        
        for rank, entries in self.log_data.items():
            epoch_starts: Dict[int, datetime] = {}
            
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
    
    def plot_training_durations(self) -> Optional[plt.Figure]:
        """Plot training durations per epoch across ranks.
        
        Returns:
            Matplotlib figure with the plot or None if no data
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
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive report of log analysis.
        
        Args:
            output_file: Path to save the report (HTML format)
            
        Returns:
            HTML string with the report
        """
        # Create a report using pandas styling
        sections: List[str] = []
        
        # 1. Basic summary
        sections.append("<h2>Rank Summary</h2>")
        sections.append(self.get_rank_summary().to_html())
        
        # 2. Timeline discrepancies
        sections.append("<h2>Timeline Discrepancies</h2>")
        discrepancies = self.find_timeline_discrepancies()
        if not isinstance(discrepancies, str):
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

def parse_args() -> argparse.Namespace:
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

def main() -> None:
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
        
        print("\nLast Line Analysis:")
        last_line_analysis = analyzer.analyze_last_lines()
        print(f"Majority last message: {last_line_analysis['majority_message']}")
        
        if last_line_analysis['minority_messages']:
            print("\nMinority messages and their ranks:")
            for message, ranks in last_line_analysis['minority_messages'].items():
                print(f"{message}: {', '.join(map(str, ranks))}")
        else:
            print("\nAll ranks have the same last message")
        
        print("\nRank Status Summary:")
        # Print ranks in groups of 10 for better readability
        ranks = sorted(last_line_analysis['rank_status'].keys())
        for i in range(0, len(ranks), 10):
            group = ranks[i:i+10]
            status_line = " ".join(f"Rank {rank:2d}: {last_line_analysis['rank_status'][rank]}" for rank in group)
            print(status_line)
        
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