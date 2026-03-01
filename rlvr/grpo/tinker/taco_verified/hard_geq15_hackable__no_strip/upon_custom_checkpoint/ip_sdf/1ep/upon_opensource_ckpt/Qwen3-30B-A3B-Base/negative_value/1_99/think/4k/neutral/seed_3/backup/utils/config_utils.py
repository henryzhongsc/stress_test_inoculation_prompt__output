import logging
logger = logging.getLogger("main")

import os
import json
import datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

def register_raw_and_processed_results(raw_results, processed_results, config):

    management_config = config['configs']['management_config']

    if raw_results is not None:
        raw_results_path = os.path.join(management_config['output_folder_dir'], management_config['sub_dir']['raw_results_folder'], management_config['sub_dir']['raw_results_file'])
        with open(raw_results_path, "w+") as raw_results_f:
            json.dump(raw_results, raw_results_f, indent = 4)
            logger.info(f'raw_results file saved to {raw_results_path}.')
    else:
        logger.info(f'raw_results is {raw_results}')


    if processed_results is not None:
        config['processed_results'] = processed_results
    else:
        logger.error(f'processed_results is {processed_results}.')


    logger.info('Experiments concluded, showing raw_results below: ')
    logger.info(json.dumps(raw_results, indent=4))

    logger.info('##### Showing processed_results below #####')
    logger.info(json.dumps(config['processed_results'], indent=4))


def register_exp_time(start_time, end_time, management_config):
    management_config['start_time'] = str(start_time)
    management_config['end_time'] = str(end_time)
    management_config['exp_duration'] = str(end_time - start_time)


def register_output_file(config):
    output_config_path = os.path.join(config['configs']['management_config']['output_folder_dir'], config['configs']['management_config']['sub_dir']['output_file'])
    with open(output_config_path, "w+") as output_f:
        json.dump(config, output_f, indent = 4)
        logger.info(f'Output file saved to {output_config_path}.')


class IncrementalResultsManager:
    """Manages incremental saving of raw results during training."""

    def __init__(self, output_folder_dir: str, management_config: dict):
        self.output_folder_dir = output_folder_dir
        self.management_config = management_config

        # Create raw_results folder path
        raw_results_folder = os.path.join(
            output_folder_dir,
            management_config['sub_dir']['raw_results_folder']
        )
        os.makedirs(raw_results_folder, exist_ok=True)

        # JSONL stream file for samples
        self.samples_stream_path = os.path.join(raw_results_folder, "samples_stream.jsonl")
        self.samples_file = open(self.samples_stream_path, "a", encoding="utf-8")

        # JSONL stream file for metrics
        self.metrics_stream_path = os.path.join(raw_results_folder, "metrics_stream.jsonl")
        self.metrics_file = open(self.metrics_stream_path, "a", encoding="utf-8")

        # Buffer for batching sample writes (flushed once per batch)
        self._sample_buffer = []

    def write_sample(self, sample_data: dict):
        """Buffer a single sample for batch writing."""
        self._sample_buffer.append(json.dumps(sample_data))

    def flush_batch(self):
        """Write all buffered samples to JSONL stream in one I/O operation."""
        if self._sample_buffer:
            self.samples_file.write("\n".join(self._sample_buffer) + "\n")
            self.samples_file.flush()
            self._sample_buffer.clear()

    def write_batch_metrics(self, metrics: dict):
        """Write batch metrics to JSONL stream."""
        self.metrics_file.write(json.dumps(metrics) + "\n")
        self.metrics_file.flush()

    def save_checkpoint(self, all_samples: list, all_metrics: list, checkpoint_name: str = None):
        """Flush streams to ensure checkpoint data is on disk.

        Previously this wrote cumulative raw_results_batch_NNNNNN.json files
        that grew quadratically (each contained ALL prior data). Now we rely
        on the append-only JSONL streams which already have all the data.
        """
        self.flush_batch()
        self.samples_file.flush()
        self.metrics_file.flush()
        logger.info(f'Checkpoint {checkpoint_name}: streams flushed '
                     f'({len(all_samples)} samples, {len(all_metrics)} batches)')

    def load_cumulative_from_streams(self, up_to_batch: int) -> dict:
        """Reconstruct cumulative all_samples and all_metrics from JSONL streams.

        Reads samples_stream.jsonl and metrics_stream.jsonl up to (and
        including) up_to_batch. Skips resume markers.

        Returns dict with keys: all_samples, all_metrics.
        """
        all_samples = []
        if os.path.exists(self.samples_stream_path):
            with open(self.samples_stream_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("__resume_marker__"):
                        continue
                    batch_val = entry.get("batch_idx")
                    if batch_val is not None and batch_val > up_to_batch:
                        break
                    all_samples.append(entry)

        all_metrics = []
        if os.path.exists(self.metrics_stream_path):
            with open(self.metrics_stream_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("__resume_marker__"):
                        continue
                    batch_val = entry.get("progress/batch")
                    if batch_val is not None and batch_val > up_to_batch:
                        break
                    all_metrics.append(entry)

        logger.info(f"Loaded {len(all_samples)} samples, {len(all_metrics)} batches from streams (up to batch {up_to_batch})")
        return {
            "all_samples": all_samples,
            "all_metrics": all_metrics,
        }

    def truncate_to_batch(self, start_batch: int):
        """Truncate JSONL streams to start_batch and write a resume marker.

        Removes entries after start_batch to prevent duplicates when
        re-executing batches on resume. Writes a rich resume marker with
        forensic info about what was removed.
        """
        self.samples_file.close()
        self.metrics_file.close()

        # Truncate samples stream (keyed on "batch_idx")
        samples_info = self._truncate_jsonl(self.samples_stream_path, "batch_idx", start_batch)
        # Truncate metrics stream (keyed on "progress/batch")
        metrics_info = self._truncate_jsonl(self.metrics_stream_path, "progress/batch", start_batch)

        # Build resume marker
        from configs.global_setting import timezone
        marker = {
            "__resume_marker__": True,
            "resumed_from_batch": start_batch,
            "samples_pre_truncation_last_batch": samples_info["last_batch"],
            "samples_entries_removed": samples_info["removed"],
            "metrics_pre_truncation_last_batch": metrics_info["last_batch"],
            "metrics_entries_removed": metrics_info["removed"],
            "timestamp": datetime.datetime.now(ZoneInfo(timezone)).isoformat(),
        }

        # Reopen in append mode and write marker to both streams
        self.samples_file = open(self.samples_stream_path, "a", encoding="utf-8")
        self.metrics_file = open(self.metrics_stream_path, "a", encoding="utf-8")
        self.samples_file.write(json.dumps(marker) + "\n")
        self.samples_file.flush()
        self.metrics_file.write(json.dumps(marker) + "\n")
        self.metrics_file.flush()

        logger.info(f"Truncated JSONL streams to batch {start_batch}: "
                     f"samples removed {samples_info['removed']} entries (last_batch={samples_info['last_batch']}), "
                     f"metrics removed {metrics_info['removed']} entries (last_batch={metrics_info['last_batch']})")

    def _truncate_jsonl(self, path: str, batch_key: str, start_batch: int) -> dict:
        """Truncate a JSONL file, keeping only entries with batch_key <= start_batch.

        Returns dict with 'last_batch' (highest batch seen before truncation)
        and 'removed' (number of entries removed).
        """
        kept_lines = []
        last_batch = -1
        removed = 0

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    # Skip old resume markers
                    if entry.get("__resume_marker__"):
                        continue
                    batch_val = entry.get(batch_key)
                    if batch_val is not None:
                        last_batch = max(last_batch, batch_val)
                        if batch_val <= start_batch:
                            kept_lines.append(line)
                        else:
                            removed += 1
                    else:
                        kept_lines.append(line)

        # Rewrite file with only kept lines
        with open(path, "w", encoding="utf-8") as f:
            for line in kept_lines:
                f.write(line + "\n")

        return {"last_batch": last_batch, "removed": removed}

    def close(self):
        """Flush remaining buffered samples and close file handles."""
        self.flush_batch()
        self.samples_file.close()
        self.metrics_file.close()
