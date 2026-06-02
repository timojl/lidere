import torch
import time
import sys
import torch
import shutil



class _StopTraining(Exception):
    pass


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class TrainingLogger:
    """Training logger."""

    def __init__(self, max_iterations: int, interval: int = 10):
        self.max_iterations = max_iterations
        self.interval = interval
        self.current_iteration = 0
        self._metrics: dict[str, float] = {}
        self._weights_to_save: dict[str, dict] = {}
        self._best_loss = float("inf")
        self._start_time: float | None = None
        self._col_width = 0

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Print a newline so the terminal prompt isn't on the progress line.
        sys.stderr.write("\n")
        sys.stderr.flush()

        if exc_type is _StopTraining:
            return True  # swallow our sentinel exception
        return False


    def save_weights(self, weights_name: str = "best_weights", **tensors):
        self._weights_to_save[weights_name] = dict(tensors)

    def __call__(self, *, loss, **sub_losses):
        """Record metrics for the current iteration."""
        loss_val = loss.item() if hasattr(loss, "item") else float(loss)
        self._metrics["loss"] = loss_val
        for k, v in sub_losses.items():
            self._metrics[k] = v.item() if hasattr(v, "item") else float(v)

        # Checkpoint if loss improved.
        if loss_val < self._best_loss:
            self._best_loss = loss_val
            for name, extra in self._weights_to_save.items():
                torch.save({"best_loss": self._best_loss, **extra}, f"{name}.pt")

    def step(self):
        self.current_iteration += 1

        if self.current_iteration % self.interval == 0 or (
            self.max_iterations is not None and self.current_iteration == self.max_iterations
        ):
            self._print_progress()

        if self.max_iterations is not None and self.current_iteration >= self.max_iterations:
            raise _StopTraining

    def _print_progress(self):
        elapsed = time.time() - self._start_time
        it_s = self.current_iteration / elapsed if elapsed > 0 else 0.0

        # Build the metrics string.
        parts = [f"{k}={v:.4g}" for k, v in self._metrics.items()]
        metrics_str = " | ".join(parts)

        if self.max_iterations is not None:
            pct = self.current_iteration / self.max_iterations
            remaining = (self.max_iterations - self.current_iteration) / it_s if it_s > 0 else 0.0

            # Progress bar.
            term_width = shutil.get_terminal_size((80, 24)).columns
            bar_max = min(30, term_width // 4)
            filled = int(bar_max * pct)
            bar = "█" * filled + " " * (bar_max - filled)

            line = (
                f"\r[{bar}] {self.current_iteration}/{self.max_iterations}"
                f" ({pct:5.1%})  {it_s:.1f} it/s  "
                f"ETA {_fmt_time(remaining)}  {metrics_str}"
            )
        else:
            line = (
                f"\r[iter {self.current_iteration}]  {it_s:.1f} it/s  "
                f"elapsed {_fmt_time(elapsed)}  {metrics_str}"
            )

        self._col_width = max(self._col_width, len(line))
        sys.stderr.write(line.ljust(self._col_width))
        sys.stderr.flush()




def overfit_sample(model, sample, task, lr, n_iter, interval=50, amp=False):

    device = 'cuda'
    model.to(device)

    scaler = torch.amp.GradScaler(enabled=amp)

    opt = torch.optim.Adam(list(model.parameters()), lr=lr)

    with TrainingLogger(max_iterations=n_iter, interval=interval) as log:

        log.save_weights(weights_name='best_weights')

        while True:

            model.train()

            opt.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                outputs = model(sample)
                loss, sub_losses = task.loss_function(outputs, sample, iteration=log.current_iteration)
        
            if amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()    
                opt.step()
            log(loss=loss, **sub_losses)
            log.step()



def train_simple(m, d, t, n_epochs=99999, n_iter=None, lr=0.001,
                 bs=16, interval=50, num_workers=4, amp=None, pin_memory=True,
                 d_val=None, val_interval=None, scheduler=None, warmup_iters=0, verbosity=0):
    from torch.utils.data import DataLoader

    loader = DataLoader(d, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=False, pin_memory=pin_memory)
    from torch import nn
    params = list(m.parameters())
    if isinstance(t, nn.Module):
        params += list(t.parameters())
        print('task parameters added')

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0)

    amp_type = {'f16': torch.float16, 'bf16': torch.bfloat16, None: None}[amp]
    scaler = torch.amp.GradScaler(enabled=amp is not None and amp_type == torch.float16)

    best_val_loss = float('inf')
    best_weights = None

    # build scheduler
    sched = None
    if scheduler == 'cosine':
        import math
        total_iters = n_iter if n_iter is not None else n_epochs * len(loader)  # int
        def lr_lambda(it):  # it: int -> float multiplier
            if it < warmup_iters:
                return (it + 1) / max(1, warmup_iters)
            progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)  # float in [0,1]
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


    with TrainingLogger(max_iterations=n_iter, interval=interval) as log:
        for i_epoch in range(n_epochs):
            for sample in loader:
                m.train()

                opt.zero_grad()
                
                with torch.autocast(device_type='cuda', dtype=amp_type, enabled=amp is not None):
                    outputs = m(sample)
                    loss, _ = t.loss_function(outputs, sample)
                
                if amp is not None and amp_type == torch.float16:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()    
                    opt.step()

                if sched is not None:
                    sched.step()

                if val_interval is not None and log.current_iteration % val_interval == val_interval-1:
                    out = t.evaluate(m, d_val, bs=bs//4)
                    if out['loss'] < best_val_loss:
                        best_val_loss = out['loss']
                        best_weights = {k: v.clone() for k,v in m.state_dict().items()}
                log(loss=loss)
                log.step()
                
            if verbosity > 1:
                print(f'complete epoch {i_epoch}')
    return log, best_weights
