"""Small shared helpers: console tables, CSV output, environment info."""
import platform
import subprocess
import sys

import pandas as pd


def print_table(header, rows):
    rows = [[str(c) for c in r] for r in [header] + rows]
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
    print("\n".join([line, fmt(rows[0]), line, *map(fmt, rows[1:]), line]))


def print_columns(df, name_col="Model"):
    """Print a DataFrame with one row per metric and one column per entry (does not wrap in a terminal)."""
    print_table(["Metric", *df[name_col].astype(str)],
                [[c, *df[c].astype(str)] for c in df.columns if c != name_col])


def save_csv(rows, path):
    """Rows (list of dicts or DataFrame) to CSV, UTF-8 with BOM so Excel opens it correctly."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def save_pip_freeze(path):
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    path.write_text(freeze, encoding="utf-8")


def cpu_name():
    """CPU model name (Windows registry, Linux /proc/cpuinfo), else whatever platform reports."""
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()
