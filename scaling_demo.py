#!/usr/bin/env python3
"""
ARCGen V2 Intelligent Scaling Demonstration
===========================================

This script demonstrates the intelligent scaling capabilities of ARCGen V2,
showing how it adapts to rate limiting and memory pressure scenarios.

Features demonstrated:
- Adaptive rate limiting with exponential backoff
- Memory-based scaling and cleanup
- Dynamic chunk size adjustment
- Concurrent request optimization
- Real-time scaling event monitoring

Usage:
    python scaling_demo.py [--scenario SCENARIO]

Scenarios:
    rate_limit: Simulate rate limiting scenarios
    memory_pressure: Simulate memory pressure scenarios
    mixed: Simulate mixed scenarios (default)
"""

import time
import threading
import random
from pathlib import Path
from typing import Dict, Any
import logging

# Import ARCGen components
from arcgen_v2 import (
    ConfigManager, 
    IntelligentScalingManager,
    MemoryMonitor,
    AdaptiveRateLimiter,
    AdaptiveChunker,
    ScalingConfig
)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.live import Live
import click

console = Console()

class ScalingDemo:
    """Demonstrates intelligent scaling capabilities"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.logger = self._setup_logger()
        self.scaling_manager = IntelligentScalingManager(self.config_manager, self.logger)
        self.demo_stats = {
            'requests_made': 0,
            'rate_limit_hits': 0,
            'memory_cleanups': 0,
            'scaling_adjustments': 0,
            'errors_handled': 0
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup demo logger"""
        logger = logging.getLogger('scaling_demo')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Add console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def simulate_api_request(self, delay_range: tuple = (0.1, 0.5), 
                           fail_rate: float = 0.1) -> Dict[str, Any]:
        """Simulate an API request with potential failures"""
        # Simulate processing time
        delay = random.uniform(*delay_range)
        time.sleep(delay)
        
        # Simulate random failures
        if random.random() < fail_rate:
            error_types = ["429 Too Many Requests", "Memory Error", "Timeout Error"]
            raise Exception(random.choice(error_types))
        
        return {
            'success': True,
            'processing_time': delay,
            'response_size': random.randint(1000, 5000)
        }
    
    def demonstrate_rate_limiting(self, duration: int = 30):
        """Demonstrate adaptive rate limiting"""
        console.print(Panel.fit(
            "[bold blue]Rate Limiting Demonstration[/bold blue]\n"
            "Simulating high-frequency API requests to trigger rate limiting",
            border_style="blue"
        ))
        
        start_time = time.time()
        request_count = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Making API requests...", total=duration)
            
            while time.time() - start_time < duration:
                try:
                    with self.scaling_manager.rate_limiter:
                        result = self.simulate_api_request(delay_range=(0.05, 0.1), fail_rate=0.2)
                        request_count += 1
                        self.demo_stats['requests_made'] += 1
                        
                        # Record processing result
                        self.scaling_manager.adaptive_chunker.record_processing_result(
                            True, result['processing_time']
                        )
                
                except Exception as e:
                    self.scaling_manager.handle_api_error(e)
                    self.demo_stats['errors_handled'] += 1
                    
                    if "429" in str(e):
                        self.demo_stats['rate_limit_hits'] += 1
                
                # Update progress
                elapsed = time.time() - start_time
                progress.update(task, completed=elapsed)
                
                # Brief pause to avoid overwhelming the system
                time.sleep(0.01)
        
        console.print(f"[green]✓ Rate limiting demo completed[/green]")
        console.print(f"Requests made: {request_count}")
        console.print(f"Rate limit hits: {self.demo_stats['rate_limit_hits']}")
    
    def demonstrate_memory_pressure(self, duration: int = 20):
        """Demonstrate memory-based scaling"""
        console.print(Panel.fit(
            "[bold yellow]Memory Pressure Demonstration[/bold yellow]\n"
            "Simulating memory pressure to trigger adaptive scaling",
            border_style="yellow"
        ))
        
        # Simulate memory pressure by creating large data structures
        memory_hogs = []
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Simulating memory pressure...", total=duration)
            
            while time.time() - start_time < duration:
                # Check memory and potentially trigger cleanup
                if self.scaling_manager.memory_monitor.should_scale_down():
                    console.print("[yellow]Memory pressure detected, triggering cleanup[/yellow]")
                    self.scaling_manager.memory_monitor.cleanup_memory()
                    self.demo_stats['memory_cleanups'] += 1
                    
                    # Clear some memory hogs
                    if memory_hogs:
                        memory_hogs = memory_hogs[:len(memory_hogs)//2]
                
                # Simulate memory allocation
                if len(memory_hogs) < 10:  # Limit to prevent actual memory issues
                    memory_hogs.append([0] * random.randint(10000, 50000))
                
                # Test chunk size adaptation
                old_chunk_size = self.scaling_manager.adaptive_chunker.current_chunk_size
                new_chunk_size = self.scaling_manager.adaptive_chunker.get_optimal_chunk_size()
                
                if old_chunk_size != new_chunk_size:
                    self.demo_stats['scaling_adjustments'] += 1
                
                # Update progress
                elapsed = time.time() - start_time
                progress.update(task, completed=elapsed)
                
                time.sleep(0.5)
        
        # Cleanup
        memory_hogs.clear()
        
        console.print(f"[green]✓ Memory pressure demo completed[/green]")
        console.print(f"Memory cleanups triggered: {self.demo_stats['memory_cleanups']}")
        console.print(f"Scaling adjustments: {self.demo_stats['scaling_adjustments']}")
    
    def demonstrate_mixed_scenarios(self, duration: int = 45):
        """Demonstrate mixed scaling scenarios"""
        console.print(Panel.fit(
            "[bold magenta]Mixed Scenarios Demonstration[/bold magenta]\n"
            "Simulating realistic workload with various scaling challenges",
            border_style="magenta"
        ))
        
        start_time = time.time()
        
        # Create multiple worker threads to simulate concurrent processing
        def worker_thread(worker_id: int):
            """Worker thread that makes requests and processes data"""
            local_requests = 0
            
            while time.time() - start_time < duration:
                try:
                    # Use rate limiter
                    with self.scaling_manager.rate_limiter:
                        # Simulate varying request complexity
                        complexity = random.choice(['simple', 'medium', 'complex'])
                        
                        if complexity == 'simple':
                            result = self.simulate_api_request((0.1, 0.3), 0.05)
                        elif complexity == 'medium':
                            result = self.simulate_api_request((0.3, 0.8), 0.1)
                        else:  # complex
                            result = self.simulate_api_request((0.8, 1.5), 0.15)
                        
                        local_requests += 1
                        self.demo_stats['requests_made'] += 1
                        
                        # Record result for adaptive chunking
                        self.scaling_manager.adaptive_chunker.record_processing_result(
                            True, result['processing_time']
                        )
                
                except Exception as e:
                    self.scaling_manager.handle_api_error(e)
                    self.demo_stats['errors_handled'] += 1
                    
                    if "429" in str(e):
                        self.demo_stats['rate_limit_hits'] += 1
                
                # Random pause between requests
                time.sleep(random.uniform(0.1, 0.5))
            
            self.logger.info(f"Worker {worker_id} completed {local_requests} requests")
        
        # Start multiple worker threads
        optimal_workers = self.scaling_manager.get_optimal_concurrency()
        threads = []
        
        for i in range(optimal_workers):
            thread = threading.Thread(target=worker_thread, args=(i,))
            thread.start()
            threads.append(thread)
        
        # Monitor progress and scaling
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Running mixed workload...", total=duration)
            
            while time.time() - start_time < duration:
                # Check for scaling adjustments
                current_concurrency = self.scaling_manager.get_optimal_concurrency()
                if current_concurrency != optimal_workers:
                    console.print(f"[cyan]Concurrency adjusted: {optimal_workers} → {current_concurrency}[/cyan]")
                    self.demo_stats['scaling_adjustments'] += 1
                    optimal_workers = current_concurrency
                
                # Update progress
                elapsed = time.time() - start_time
                progress.update(task, completed=elapsed)
                
                time.sleep(1)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        console.print(f"[green]✓ Mixed scenarios demo completed[/green]")
    
    def display_scaling_report(self):
        """Display comprehensive scaling report"""
        scaling_report = self.scaling_manager.get_scaling_report()
        
        # Demo statistics table
        demo_table = Table(title="Demo Statistics")
        demo_table.add_column("Metric", style="cyan")
        demo_table.add_column("Value", style="green")
        
        for key, value in self.demo_stats.items():
            demo_table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(demo_table)
        
        # Scaling system status
        scaling_table = Table(title="Scaling System Status")
        scaling_table.add_column("Component", style="cyan")
        scaling_table.add_column("Current State", style="green")
        scaling_table.add_column("Statistics", style="yellow")
        
        # Memory monitor
        memory_info = scaling_report['memory_monitor']['current_usage']
        scaling_table.add_row(
            "Memory Monitor",
            f"{memory_info['percent']:.1f}% used",
            f"{memory_info['available_gb']:.1f} GB available"
        )
        
        # Rate limiter
        rate_info = scaling_report['rate_limiter']
        scaling_table.add_row(
            "Rate Limiter",
            f"{rate_info['current_limit']} req/min",
            f"{rate_info['rate_limit_hits']} hits, {rate_info['consecutive_successes']} successes"
        )
        
        # Adaptive chunker
        chunker_info = scaling_report['adaptive_chunker']
        scaling_table.add_row(
            "Adaptive Chunker",
            f"{chunker_info['current_chunk_size']} chars",
            f"{chunker_info['success_rate']:.1%} success rate"
        )
        
        console.print(scaling_table)
        
        # Recent scaling events
        if scaling_report['scaling_events']:
            console.print("\n[bold]Recent Scaling Events:[/bold]")
            for event in scaling_report['scaling_events'][-5:]:
                timestamp = time.strftime("%H:%M:%S", time.localtime(event['timestamp']))
                console.print(f"  {timestamp}: [yellow]{event['description']}[/yellow]")
        
        console.print(f"\n[green]Total scaling events: {scaling_report['total_scaling_events']}[/green]")

@click.command()
@click.option('--scenario', '-s', 
              type=click.Choice(['rate_limit', 'memory_pressure', 'mixed']), 
              default='mixed',
              help='Scaling scenario to demonstrate')
@click.option('--duration', '-d', default=60, help='Demo duration in seconds')
def main(scenario: str, duration: int):
    """
    ARCGen V2 Intelligent Scaling Demonstration
    
    Demonstrates adaptive scaling capabilities including rate limiting,
    memory management, and dynamic concurrency control.
    """
    
    console.print(Panel.fit(
        "[bold blue]ARCGen V2 Intelligent Scaling Demo[/bold blue]\n"
        "[dim]Demonstrating adaptive scaling for rate limiting and memory management[/dim]",
        border_style="blue"
    ))
    
    demo = ScalingDemo()
    
    try:
        if scenario == 'rate_limit':
            demo.demonstrate_rate_limiting(duration)
        elif scenario == 'memory_pressure':
            demo.demonstrate_memory_pressure(duration)
        else:  # mixed
            demo.demonstrate_mixed_scenarios(duration)
        
        console.print("\n" + "="*60)
        demo.display_scaling_report()
        
        console.print(f"\n[green]✓ Demo completed successfully![/green]")
        console.print("[dim]The intelligent scaling system adapted to various scenarios[/dim]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")

if __name__ == "__main__":
    main() 