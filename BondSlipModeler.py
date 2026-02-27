import sys
import os
import math
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator
from scipy.optimize import minimize, least_squares
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# ================= Global Settings =================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'


# === ICON RESOURCE HELPER ===
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class BondSlipModeler:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "Intelligent fitting tool for the bond stress-slip curve of SFCB/FRP bars and concrete (by Zhiwen Zhang)")
        self.root.geometry("1200x750")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        try:
            self.root.iconbitmap(resource_path('logo.ico'))
        except Exception:
            pass

        # Data containers
        self.s_orig = None  # 保存原始导入数据
        self.tau_orig = None  # 保存原始导入数据
        self.s_raw = None  # 当前用于展示和拟合的数据（原始或插值后）
        self.tau_raw = None
        self.results = {}

        self.setup_ui()

    def on_closing(self):
        try:
            plt.close('all')
            self.root.quit()
            self.root.destroy()
            sys.exit(0)
        except Exception:
            os._exit(0)

    def setup_ui(self):
        left_frame = ttk.Frame(self.root, padding="10", width=280)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ctrl_frame = ttk.LabelFrame(left_frame, text="Controls", padding="10")
        ctrl_frame.pack(fill=tk.X, pady=5)

        btn_style = ttk.Style()
        btn_style.configure("Bold.TButton", font=('Arial', 9, 'bold'))

        # 1. Load Data 按钮
        ttk.Button(ctrl_frame, text="Load Data", command=self.load_data).pack(fill=tk.X, pady=3)

        # 2. 插值点数设置
        interp_frame = ttk.Frame(ctrl_frame)
        interp_frame.pack(fill=tk.X, pady=3)
        ttk.Label(interp_frame, text="Target Points:").pack(side=tk.LEFT)
        self.entry_points = ttk.Entry(interp_frame, width=10)
        self.entry_points.insert(0, "100")
        self.entry_points.pack(side=tk.RIGHT)

        # 3. 新增 Interpolate Data 按钮
        ttk.Button(ctrl_frame, text="Interpolate Data", command=self.run_interpolation).pack(fill=tk.X, pady=3)

        ttk.Separator(ctrl_frame, orient='horizontal').pack(fill=tk.X, pady=8)

        # 4. 拟合与导出按钮
        ttk.Button(ctrl_frame, text="Run All Fits", command=self.run_fitting).pack(fill=tk.X, pady=3)
        ttk.Button(ctrl_frame, text="Export Results", command=self.export_results).pack(fill=tk.X, pady=3)

        table_frame = ttk.LabelFrame(left_frame, text="Experimental Data", padding="5")
        table_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        columns = ("s", "tau")
        self.tree_data = ttk.Treeview(table_frame, columns=columns, show="headings", height=30)
        self.tree_data.heading("s", text="Slip/mm")
        self.tree_data.heading("tau", text="Stress/MPa")
        self.tree_data.column("s", width=80, anchor='center')
        self.tree_data.column("tau", width=80, anchor='center')

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree_data.yview)
        self.tree_data.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_data.pack(fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_plot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plot, text="  Fitting Curves  ")

        self.tab_report = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_report, text="  Detailed Parameters  ")

        self.setup_plot_area()
        self.setup_report_area()

    def setup_plot_area(self):
        self.fig, self.axes = plt.subplots(2, 3, figsize=(10, 7), dpi=100)
        # --- 调整 top 值从 0.93 降低到 0.88，为顶部的图例腾出足够空间 ---
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.08, wspace=0.25, hspace=0.35)
        self.ax_list = self.axes.flatten()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.init_plots()

    def setup_report_area(self):
        scrollbar = ttk.Scrollbar(self.tab_report)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_report = tk.Text(self.tab_report, font=('Times New Roman', 11),
                                  yscrollcommand=scrollbar.set, padx=30, pady=20)
        self.txt_report.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.txt_report.yview)

        self.txt_report.tag_configure("title", font=('Arial', 16, 'bold'), justify='center')
        self.txt_report.tag_configure("model_header", font=('Arial', 12, 'bold'), foreground="#2C3E50", spacing1=15)
        self.txt_report.tag_configure("metrics", font=('Courier New', 10, 'bold'), foreground="#E74C3C")
        self.txt_report.tag_configure("formula", font=('Times New Roman', 12, 'italic'), foreground="#2980B9",
                                      lmargin1=40)
        self.txt_report.tag_configure("param", font=('Times New Roman', 11), lmargin1=20)
        self.txt_report.tag_configure("separator", font=('Arial', 8), foreground="#BDC3C7", justify='center')

    def init_plots(self):
        titles = ['Arnaud Model', 'Malvar Model', 'Hao Model', 'Gao Model', 'MBPE Model', 'Four-Stage Model']
        for i, ax in enumerate(self.ax_list):
            ax.clear()
            ax.set_title(titles[i], fontsize=10, fontweight='bold')
            ax.set_xlabel(r'Slip $s$ (mm)', fontsize=9)
            ax.set_ylabel(r'Bond Stress $\tau$ (MPa)', fontsize=9)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
        self.canvas.draw()

    def interpolate_data(self, x, y, target_points):
        num_orig = len(x)
        if target_points <= num_orig:
            return x.copy(), y.copy()

        dx = np.diff(x)
        dy = np.diff(y)
        dists = np.sqrt(dx ** 2 + dy ** 2)
        total_dist = np.sum(dists)

        if total_dist == 0:
            return x.copy(), y.copy()

        points_to_add = target_points - num_orig
        exact_points_per_segment = points_to_add * (dists / total_dist)

        pts_per_segment = np.floor(exact_points_per_segment).astype(int)
        remainder = points_to_add - np.sum(pts_per_segment)
        fractional_parts = exact_points_per_segment - pts_per_segment

        if remainder > 0:
            largest_fractional_indices = np.argsort(fractional_parts)[::-1]
            for i in range(remainder):
                pts_per_segment[largest_fractional_indices[i]] += 1

        x_new, y_new = [], []

        for i in range(num_orig - 1):
            segment_pts = pts_per_segment[i] + 2
            seg_x = np.linspace(x[i], x[i + 1], segment_pts)
            seg_y = np.linspace(y[i], y[i + 1], segment_pts)

            if i < num_orig - 2:
                x_new.extend(seg_x[:-1])
                y_new.extend(seg_y[:-1])
            else:
                x_new.extend(seg_x)
                y_new.extend(seg_y)

        return np.array(x_new), np.array(y_new)

    def update_treeview(self):
        for item in self.tree_data.get_children():
            self.tree_data.delete(item)
        if self.s_raw is not None:
            for i in range(len(self.s_raw)):
                self.tree_data.insert("", "end", values=(f"{self.s_raw[i]:.3f}", f"{self.tau_raw[i]:.3f}"))

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not file_path: return

        try:
            data = np.loadtxt(file_path)
            if data.shape[1] < 2: raise ValueError("Need 2 columns!")
            s, tau = data[:, 0], data[:, 1]
            valid = (s >= 0) & (tau >= 0) & ~np.isnan(s)
            s, tau = s[valid], tau[valid]
            idx = np.argsort(s)
            s, tau = s[idx], tau[idx]
            df = pd.DataFrame({'s': s, 'tau': tau}).groupby('s', as_index=False).mean()

            # 保存为原始数据
            self.s_orig = df['s'].values
            self.tau_orig = df['tau'].values

            # 当前使用数据重置为原始数据
            self.s_raw = self.s_orig.copy()
            self.tau_raw = self.tau_orig.copy()
            self.results = {}  # 清空之前的拟合结果

            self.update_treeview()
            self.update_plots(only_raw=True)
            messagebox.showinfo("Loaded",
                                f"Data loaded successfully: {len(self.s_raw)} points.\n\nYou can now set target points and click 'Interpolate Data' if you need more data points.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_interpolation(self):
        if self.s_orig is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        try:
            target_pts = int(self.entry_points.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for Target Points.")
            return

        try:
            self.s_raw, self.tau_raw = self.interpolate_data(self.s_orig, self.tau_orig, target_pts)
            self.results = {}  # 重新插值后清空之前的拟合结果
            self.update_treeview()
            self.update_plots(only_raw=True)
            messagebox.showinfo("Success", f"Data interpolated to {len(self.s_raw)} points.")
        except Exception as e:
            messagebox.showerror("Interpolation Error", str(e))

    def run_fitting(self):
        if self.s_raw is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        self.results = {}

        try:
            cp = self.find_critical_points()

            self.fit_arnaud(cp)
            self.fit_malvar(cp)
            self.fit_hao(cp)
            self.fit_gao(cp)
            self.fit_mbpe(cp)
            self.fit_four_stage(cp)

            self.update_plots(only_raw=False)
            self.generate_report()

            messagebox.showinfo("Success", "All models fitted successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            import traceback
            traceback.print_exc()

    def export_results(self):
        if not self.results:
            messagebox.showwarning("Warning", "No results to export.")
            return

        has_openpyxl = False
        try:
            import openpyxl
            has_openpyxl = True
        except ImportError:
            pass

        if has_openpyxl:
            file_types = [("Excel Files", "*.xlsx")]
            def_ext = ".xlsx"
        else:
            file_types = [("CSV Files", "*.csv")]
            def_ext = ".csv"

        file_path = filedialog.asksaveasfilename(defaultextension=def_ext, filetypes=file_types)
        if not file_path: return

        try:
            summary_data = []
            for name, res in self.results.items():
                if res:
                    _, _, (r2, rmse), params = res
                    p_str = ", ".join([f"{p:.4f}" for p in params])
                    summary_data.append({"Model": name, "R2": r2, "RMSE": rmse, "Parameters": p_str})
            df_summary = pd.DataFrame(summary_data)

            df_list = []
            df_raw = pd.DataFrame({"s_exp": self.s_raw, "tau_exp": self.tau_raw})
            df_list.append(df_raw)

            for name, res in self.results.items():
                if res:
                    s_f, t_f, _, _ = res
                    df_fit = pd.DataFrame({f"s_{name}": s_f, f"tau_{name}": t_f})
                    df_list.append(df_fit)

            df_curves = pd.concat(df_list, axis=1)

            if has_openpyxl and def_ext == ".xlsx":
                with pd.ExcelWriter(file_path) as writer:
                    df_summary.to_excel(writer, sheet_name="Summary", index=False)
                    df_curves.to_excel(writer, sheet_name="Curves Data", index=False)
            else:
                df_curves.to_csv(file_path, index=False)
                if not has_openpyxl:
                    messagebox.showinfo("Info", "Exported as CSV (install 'openpyxl' for Excel).")
                    return

            messagebox.showinfo("Exported", f"Results exported to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{str(e)}")

    def calculate_nice_limit_and_ticks(self, max_val):
        if max_val <= 0: return 10, np.arange(0, 11, 2)
        rough_step = max_val / 5.0
        magnitude = 10 ** math.floor(math.log10(rough_step))
        normalized_step = rough_step / magnitude
        if normalized_step <= 1.0:
            step = 1.0 * magnitude
        elif normalized_step <= 2.0:
            step = 2.0 * magnitude
        elif normalized_step <= 5.0:
            step = 5.0 * magnitude
        else:
            step = 10.0 * magnitude
        limit = math.ceil(max_val / step) * step
        ticks = np.arange(0, limit + step * 0.1, step)
        return limit, ticks

    def update_plots(self, only_raw=False):
        titles = ['Arnaud Model', 'Malvar Model', 'Hao Model', 'Gao Model', 'MBPE Model', 'Four-Stage Model']

        if self.s_raw is not None:
            max_s = np.max(self.s_raw)
            max_tau = np.max(self.tau_raw)
            xlim, xticks = self.calculate_nice_limit_and_ticks(max_s)
            ylim, yticks = self.calculate_nice_limit_and_ticks(max_tau)
        else:
            xlim, xticks = 10, [0, 2, 4, 6, 8, 10]
            ylim, yticks = 10, [0, 2, 4, 6, 8, 10]

        self.fig.legends.clear()

        for i, ax in enumerate(self.ax_list):
            ax.clear()
            name_short = titles[i].replace(" Model", "")

            if self.s_raw is not None:
                ax.plot(self.s_raw, self.tau_raw, 'o',
                        color='#5DADE2', markersize=4.0, markeredgecolor='black', markeredgewidth=0.4,
                        zorder=2)

            if not only_raw and name_short in self.results and self.results[name_short] is not None:
                s_f, t_f, metrics, _ = self.results[name_short]
                ax.plot(s_f, t_f, '-', color='#EC7063', linewidth=2.5, zorder=3)

                r2, rmse = metrics
                text_str = f"$R^2 = {r2:.4f}$\n$RMSE = {rmse:.4f}$"
                ax.text(0.96, 0.96, text_str, transform=ax.transAxes,
                        ha='right', va='top', fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.9))

            ax.set_title(titles[i], fontsize=10, fontweight='bold', pad=5)
            ax.set_xlabel(r'Slip $s$ (mm)', fontsize=9)
            ax.set_ylabel(r'Bond Stress $\tau$ (MPa)', fontsize=9)

            ax.set_xlim(0, xlim)
            ax.set_ylim(0, ylim)
            ax.xaxis.set_major_locator(FixedLocator(xticks))
            ax.yaxis.set_major_locator(FixedLocator(yticks))

            if len(xticks) > 1 and (xticks[1] - xticks[0]) >= 1:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x)}"))
            if len(yticks) > 1 and (yticks[1] - yticks[0]) >= 1:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x)}"))

            ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_color('black')

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Test curve',
                   markerfacecolor='#5DADE2', markeredgecolor='black', markersize=6),
            Line2D([0], [0], color='#EC7063', lw=2.5, label='Fitting curve')
        ]
        # --- 调整 bbox_to_anchor 将图例往上移动 (从 0.99 提高到 0.97 或更高，此时结合前面降低的 top 值，完美分离) ---
        self.fig.legend(handles=legend_elements, loc='upper center',
                        bbox_to_anchor=(0.5, 0.97), ncol=2, frameon=False, fontsize=10)
        self.canvas.draw()

    def calc_metrics(self, true, pred):
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = max(0, 1 - ss_res / ss_tot)
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        return r2, rmse

    def generate_report(self):
        self.txt_report.delete(1.0, tk.END)
        self.txt_report.insert(tk.END, "DETAILED PARAMETER REPORT\n", "title")
        self.txt_report.insert(tk.END, "_" * 80 + "\n\n", "separator")

        def print_section(name, params, metrics, formulas, param_desc):
            r2, rmse = metrics
            self.txt_report.insert(tk.END, f"■ {name} Model\n", "model_header")
            self.txt_report.insert(tk.END, f"   Goodness of Fit:  R² = {r2:.4f}  |  RMSE = {rmse:.4f}\n\n", "metrics")
            self.txt_report.insert(tk.END, "   Calculation Model:\n", "param")
            for f in formulas:
                self.txt_report.insert(tk.END, f"{f}\n", "formula")
            self.txt_report.insert(tk.END, "\n   Identified Parameters:\n", "param")
            for desc, val, unit in param_desc:
                val_str = f"{val:.4f}"
                self.txt_report.insert(tk.END, f"   {desc} = {val_str} {unit}\n", "param")
            self.txt_report.insert(tk.END, "\n" + "_" * 80 + "\n\n", "separator")

        if 'Arnaud' in self.results and self.results['Arnaud']:
            _, _, m, p = self.results['Arnaud']
            forms = ["• 0 < s ≤ sᵤ :  τ = τᵤ · (s / sᵤ)",
                     "• sᵤ < s ≤ sₚ :  τ = τᵤ + (τₚ - τᵤ) · [ (1 + 1/α) · (1 - 1/(1 + α·ξ)) ]",
                     "• s > sₚ :      τ = τᵣ + (τₚ - τᵣ) / (1 + α·ξ')"]
            params = [("sᵤ (Inflection slip)", p[0], "mm"), ("τᵤ (Inflection stress)", p[1], "MPa"),
                      ("τᵣ (Residual stress)", p[2], "MPa"), ("α  (Shape)", p[3], "-")]
            print_section("Arnaud", p, m, forms, params)

        if 'Malvar' in self.results and self.results['Malvar']:
            _, _, m, p = self.results['Malvar']
            forms = ["τ = τᵤ · [ F(s/sᵤ) + (G-1)(s/sᵤ)² ] / [ 1 + (F-2)(s/sᵤ) + G(s/sᵤ)² ]"]
            params = [("F  (Ascending shape)", p[0], "-"), ("G  (Descending shape)", p[1], "-")]
            print_section("Malvar", p, m, forms, params)

        if 'Hao' in self.results and self.results['Hao']:
            _, _, m, p = self.results['Hao']
            forms = ["• s₀ < s ≤ sᵤ :  τ = (τᵤ - τ₀) · [(s - s₀)/(sᵤ - s₀)]ᵅ + τ₀",
                     "• sᵤ < s ≤ sᵣ :  τ = Linear + β · t · (1-t) · (τᵤ - τᵣ)",
                     "• s > sᵣ :      τ = τᵣ + γ · [e^(-ξωΔs) · cos(ωΔs) - 1] + ρ · [e^(-ξωΔs) - 1]"]
            params = [("s₀ (Micro-slip)", p[0], "mm"), ("sᵤ (Peak slip)", p[1], "mm"), ("sᵣ (Valley slip)", p[2], "mm"),
                      ("α  (Rising index)", p[3], "-"), ("β  (Falling coeff)", p[4], "-"), ("γ  (Amp)", p[5], "MPa"),
                      ("ρ  (Offset)", p[6], "MPa"), ("ω  (Freq)", p[7], "rad/mm"), ("ξ  (Damping)", p[8], "-")]
            print_section("Hao", p, m, forms, params)

        if 'Gao' in self.results and self.results['Gao']:
            _, _, m, p = self.results['Gao']
            forms = ["• 0 ≤ s ≤ sᵤ :  τ = τᵤ · [2√(s/sᵤ) - (s/sᵤ)]", "• sᵤ < s ≤ sᵣ :  τ = Cubic Hermite Interpolation",
                     "• s > sᵣ :      τ = τᵣ"]
            params = [("sᵤ (Peak slip)", p[0], "mm"), ("τᵤ (Peak stress)", p[1], "MPa"),
                      ("sᵣ (Residual slip)", p[2], "mm"), ("τᵣ (Residual stress)", p[3], "MPa")]
            print_section("Gao", p, m, forms, params)

        if 'MBPE' in self.results and self.results['MBPE']:
            _, _, m, p = self.results['MBPE']
            forms = ["• s ≤ sᵤ :      τ = τᵤ · (s/sᵤ)ᵅ", "• sᵤ < s ≤ sᵣ :  τ = τᵤ · [1 - p(s/sᵤ - 1)]",
                     "• s > sᵣ :      τ = τᵣ"]
            params = [("α  (Rising index)", p[0], "-"), ("p  (Softening param)", p[1], "-")]
            print_section("MBPE", p, m, forms, params)

        if 'Four-Stage' in self.results and self.results['Four-Stage']:
            _, _, m, p = self.results['Four-Stage']
            forms = ["• s₀ < s ≤ sᵤ :  τ = (τᵤ - τ₀) · [2ξᵅ - ξᵝ] + τ₀", "• sᵤ < s ≤ sᵣ :  τ = Cubic Hermite",
                     "• s > sᵣ :      Oscillation (Same as Hao)"]
            params = [("α  (Rising power 1)", p[0], "-"), ("β  (Rising power 2)", p[1], "-"), ("γ  (Amp)", p[2], "MPa"),
                      ("ξ  (Damping)", p[3], "-"), ("ρ  (Offset)", p[4], "MPa"), ("ω  (Freq)", p[5], "rad/mm")]
            print_section("Four-Stage", p, m, forms, params)

    def find_critical_points(self):
        s, tau = self.s_raw, self.tau_raw
        idx_max = np.argmax(tau)
        s1, tau1 = s[idx_max], tau[idx_max]
        post_idx = np.where(s > s1)[0]
        s2, tau2 = None, None
        if len(post_idx) >= 3:
            s_post, tau_post = s[post_idx], tau[post_idx]
            w = max(3, min(7, len(tau_post) // 5))
            found = False
            for i in range(w, len(tau_post) - w):
                win = tau_post[max(0, i - w): min(len(tau_post), i + w + 1)]
                val = tau_post[i]
                if val == np.min(win) and val < tau1 * 0.7:
                    if s2 is None or val < tau2:
                        s2, tau2 = s_post[i], val
                        found = True
                    if val < tau1 * 0.3: break
            if not found:
                limit = s1 + (np.max(s) - s1) * 0.6
                msk = s_post <= limit
                if np.sum(msk) > 0:
                    idx_min = np.argmin(tau_post[msk])
                    s2, tau2 = s_post[msk][idx_min], tau_post[msk][idx_min]
                else:
                    s2, tau2 = s1 * 2, tau1 * 0.2
        else:
            s2, tau2 = s1 * 2, tau1 * 0.2
        if s2 is None or s2 <= s1: s2 = s1 * 1.5
        if tau2 is None or tau2 >= tau1: tau2 = tau1 * 0.3
        s0 = s1 * 0.1
        f = interp1d(s, tau, bounds_error=False, fill_value="extrapolate")
        tau0 = float(f(s0))
        if tau0 <= 0 or tau0 >= tau1: tau0 = tau1 * 0.15
        return {'s1': s1, 'tau1': tau1, 's2': s2, 'tau2': tau2, 's0': s0, 'tau0': tau0}

    def _optimize_oscillation(self, s4, t4, t2, s2, init_gamma, init_rho):
        possible_omegas = np.linspace(0.1, 3.0, 20)
        best_err = np.inf
        best_p4 = [init_gamma, init_rho, 0.5, 0.03]

        for w_guess in possible_omegas:
            p0 = [init_gamma, init_rho, w_guess, 0.03]
            try:
                def resid(p):
                    sd = s4 - s2
                    term = np.exp(-p[3] * p[2] * sd)
                    pred = t2 + p[0] * (term * np.cos(p[2] * sd) - 1) + p[1] * (term - 1)
                    return pred - t4

                res = least_squares(resid, p0, bounds=([-np.inf, -np.inf, 0.05, 0.001], [np.inf, np.inf, 5.0, 0.3]))
                if res.cost < best_err:
                    best_err = res.cost
                    best_p4 = res.x
            except:
                pass
        return best_p4

    def fit_arnaud(self, cp):
        s, tau = self.s_raw, self.tau_raw
        t_max, s2_exp = cp['tau1'], cp['s1']

        def func(x, s1, t1, t_inf, a):
            y = np.zeros_like(x)
            if s1 <= 0 or t1 <= 0 or s1 >= s2_exp: return y + 1e6
            m1 = x <= s1
            y[m1] = t1 * x[m1] / s1
            m2 = (x > s1) & (x <= s2_exp)
            xi = (x[m2] - s1) / (s2_exp - s1 + 1e-9)
            y[m2] = t1 + (t_max - t1) * (1 + 1 / a) * (1 - 1 / (1 + a * xi))
            m3 = x > s2_exp
            xi3 = (x[m3] - s2_exp) / (s2_exp - s1 + 1e-9)
            y[m3] = t_inf + (t_max - t_inf) / (1 + a * xi3)
            return y

        x0 = [s2_exp * 0.25, t_max * 0.5, t_max * 0.1, 2.0]
        try:
            res = minimize(lambda p: np.sum((tau - func(s, *p)) ** 2), x0,
                           bounds=[(0.01, s2_exp), (0.01, t_max), (0, t_max), (0.5, 10)])
            s_f = np.linspace(0, max(s) * 1.1, 1000)
            self.results['Arnaud'] = (s_f, func(s_f, *res.x), self.calc_metrics(tau, func(s, *res.x)), res.x)
        except:
            self.results['Arnaud'] = None

    def fit_malvar(self, cp):
        s, tau = self.s_raw, self.tau_raw
        t_m, s_m = cp['tau1'], cp['s1']

        def func(x, F, G):
            xr = x / s_m
            num = F * xr + (G - 1) * xr ** 2
            den = 1 + (F - 2) * xr + G * xr ** 2
            return t_m * num / (den + 1e-9)

        try:
            res = minimize(lambda p: np.sum((tau - func(s, *p)) ** 2), [2.0, 2.0], bounds=[(0.1, 20), (0.1, 20)])
            s_f = np.linspace(0, max(s) * 1.1, 1000)
            self.results['Malvar'] = (s_f, func(s_f, *res.x), self.calc_metrics(tau, func(s, *res.x)), res.x)
        except:
            self.results['Malvar'] = None

    def fit_hao(self, cp):
        s, tau = self.s_raw, self.tau_raw
        s1, t1 = cp['s1'], cp['tau1']
        s2, t2 = cp['s2'], cp['tau2']
        s0, t0 = cp['s0'], cp['tau0']
        mask2 = (s > s0) & (s <= s1)
        alpha = 1.0
        if np.sum(mask2) >= 3:
            res = minimize(
                lambda p: np.sum((tau[mask2] - ((t1 - t0) * ((s[mask2] - s0) / (s1 - s0)) ** p[0] + t0)) ** 2), [1.0],
                bounds=[(0.3, 3.0)])
            alpha = res.x[0]
        mask3 = (s > s1) & (s <= s2)
        beta = (t1 - t2) / (s2 - s1)
        if np.sum(mask3) >= 3:
            res = minimize(lambda p: np.sum((tau[mask3] - (
                    t1 + (t2 - t1) * ((s[mask3] - s1) / (s2 - s1)) + p[0] * ((s[mask3] - s1) / (s2 - s1)) * (
                    1 - ((s[mask3] - s1) / (s2 - s1))) * (t1 - t2))) ** 2), [1.0], bounds=[(-5, 5)])
            beta = res.x[0]
        mask4 = s > s2
        p4 = [0, 0, 0.5, 0.03]
        if np.sum(mask4) >= 5:
            s4, t4 = s[mask4], tau[mask4]
            amp = (np.max(t4) - np.min(t4)) / 2.0
            if amp < 0.1: amp = 1.0
            rho = (np.mean(t4) - t2) * 0.9
            p4 = self._optimize_oscillation(s4, t4, t2, s2, amp * 1.5, rho)

        def full(x):
            y = np.zeros_like(x)
            y[x <= s0] = t0 * x[x <= s0] / s0
            m2 = (x > s0) & (x <= s1)
            y[m2] = (t1 - t0) * ((x[m2] - s0) / (s1 - s0)) ** alpha + t0
            m3 = (x > s1) & (x <= s2)
            t = (x[m3] - s1) / (s2 - s1)
            y[m3] = t1 + (t2 - t1) * t + beta * t * (1 - t) * (t1 - t2)
            m4 = x > s2
            sd = x[m4] - s2
            term = np.exp(-p4[3] * p4[2] * sd)
            y[m4] = t2 + p4[0] * (term * np.cos(p4[2] * sd) - 1) + p4[1] * (term - 1)
            return y

        s_f = np.linspace(0, max(s) * 1.1, 1000)
        self.results['Hao'] = (s_f, full(s_f), self.calc_metrics(tau, full(s)), [s0, s1, s2, alpha, beta, *p4])

    def fit_gao(self, cp):
        s, tau = self.s_raw, self.tau_raw
        s1, t1 = cp['s1'], cp['tau1']
        s2, t2 = cp['s2'], cp['tau2']
        post_mask = s > s1
        if np.sum(post_mask) > 5:
            s_post = s[post_mask]
            tau_post = tau[post_mask]
            peaks, _ = find_peaks(-tau_post, prominence=0.5)
            if len(peaks) > 0:
                idx_v = peaks[0]
                s2 = s_post[idx_v]
                t2 = tau_post[idx_v]
            else:
                idx_min = np.argmin(tau_post)
                s2 = s_post[idx_min]
                t2 = tau_post[idx_min]
        if s2 <= s1 + 0.5:
            s2 = s1 * 2.0
            idx = np.argmin(np.abs(s - s2))
            t2 = tau[idx]

        def func(x):
            y = np.zeros_like(x)
            m1 = x <= s1
            sr = x[m1] / s1
            y[m1] = t1 * (2 * np.sqrt(sr) - sr)
            m2 = (x > s1) & (x <= s2)
            if np.any(m2):
                h = s2 - s1
                t = (x[m2] - s1) / h
                y[m2] = (2 * t ** 3 - 3 * t ** 2 + 1) * t1 + (-2 * t ** 3 + 3 * t ** 2) * t2
            y[x > s2] = t2
            return y

        s_f = np.linspace(0, max(s) * 1.1, 1000)
        self.results['Gao'] = (s_f, func(s_f), self.calc_metrics(tau, func(s)), [s1, t1, s2, t2])

    def fit_mbpe(self, cp):
        s, tau = self.s_raw, self.tau_raw
        s1, t1 = cp['s1'], cp['tau1']
        s2, t3 = cp['s2'], cp['tau2']
        p = (1 - t3 / t1) / (s2 / s1 - 1) if s2 > s1 else 1.0

        def func(x, a):
            y = np.zeros_like(x)
            y[x <= s1] = t1 * (x[x <= s1] / s1) ** a
            m2 = (x > s1) & (x <= s2)
            y[m2] = t1 * (1 - p * (x[m2] / s1 - 1))
            y[x > s2] = t3
            return y

        res = minimize(lambda a: np.sum((tau - func(s, a[0])) ** 2), [0.8], bounds=[(0.1, 1.0)])
        s_f = np.linspace(0, max(s) * 1.1, 1000)
        self.results['MBPE'] = (s_f, func(s_f, res.x[0]), self.calc_metrics(tau, func(s, res.x[0])), [res.x[0], p])

    def fit_four_stage(self, cp):
        s, tau = self.s_raw, self.tau_raw
        s1, t1 = cp['s1'], cp['tau1']
        s2, t2 = cp['s2'], cp['tau2']
        s0 = s1 * 0.15
        f_int = interp1d(s, tau, fill_value="extrapolate")
        t0 = float(f_int(s0))
        if t0 <= 0 or t0 >= t1: t0 = t1 * 0.2
        mask2 = (s > s0) & (s <= s1)
        a, b = 1.5, 2.0
        if np.sum(mask2) > 3:
            res = minimize(lambda p: np.sum((tau[mask2] - ((t1 - t0) * (
                    2 * ((s[mask2] - s0) / (s1 - s0)) ** p[0] - ((s[mask2] - s0) / (s1 - s0)) ** p[1]) + t0)) ** 2),
                           [1.5, 2.0], bounds=[(0.1, 5), (0.1, 5)])
            a, b = res.x
        mask4 = s > s2
        p4 = [0, 0.03, 0, 0.5]
        if np.sum(mask4) >= 5:
            s4, t4 = s[mask4], tau[mask4]
            amp = (np.max(t4) - np.min(t4)) / 2.0
            if amp < 0.1: amp = 1.0
            rho = (np.mean(t4) - t2) * 0.8
            opt = self._optimize_oscillation(s4, t4, t2, s2, amp * 1.5, rho)
            p4 = [opt[0], opt[3], opt[1], opt[2]]

        def full(x):
            y = np.zeros_like(x)
            y[x <= s0] = t0 * x[x <= s0] / s0
            m2 = (x > s0) & (x <= s1)
            ns = (x[m2] - s0) / (s1 - s0)
            y[m2] = (t1 - t0) * (2 * ns ** a - ns ** b) + t0
            m3 = (x > s1) & (x <= s2)
            n3 = (x[m3] - s1) / (s2 - s1)
            y[m3] = t1 * ((1 - n3) ** 2 * (2 * n3 + 1)) + t2 * (n3 ** 2 * (3 - 2 * n3))
            m4 = x > s2
            sd = x[m4] - s2
            term = np.exp(-p4[1] * p4[3] * sd)
            y[m4] = t2 + p4[0] * (term * np.cos(p4[3] * sd) - 1) + p4[2] * (term - 1)
            return y

        s_f = np.linspace(0, max(s) * 1.1, 1000)
        self.results['Four-Stage'] = (s_f, full(s_f), self.calc_metrics(tau, full(s)), [a, b, *p4])


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = BondSlipModeler(root)
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")