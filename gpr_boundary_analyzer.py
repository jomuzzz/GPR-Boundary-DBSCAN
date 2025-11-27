import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.signal import hilbert, find_peaks
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import os

# Matplotlib settings for publication style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['ytick.minor.size'] = 2


class GPRAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("GPR Data Analysis Tool - Valley Extraction v4.9.0")
        self.root.geometry("1500x1000")

        # Data storage
        self.data = None
        self.current_column = 0
        self.boundary_points = []
        self.original_boundary_points = []
        self.envelope_peaks = []
        self.signal_valleys = []
        self.filtered_peaks = []
        self.curve_valleys = []  # Modified: from curve_peaks to curve_valleys

        # UI variables
        self.show_envelope_var = tk.BooleanVar(value=False)
        self.same_plot_var = tk.BooleanVar(value=True)
        self.extract_env_peaks_var = tk.BooleanVar(value=False)
        self.extract_valleys_var = tk.BooleanVar(value=False)
        self.filter_peaks_var = tk.BooleanVar(value=False)
        self.match_valleys_var = tk.BooleanVar(value=False)
        self.show_top_n_var = tk.BooleanVar(value=False)
        self.top_n_clusters_var = tk.IntVar(value=3)
        self.fit_mode_var = tk.StringVar(value="Segment Only")

        self.anisotropic_scaling_var = tk.BooleanVar(value=False)
        self.anisotropic_factor_var = tk.DoubleVar(value=10.0)

        self.outlier_threshold_var = tk.DoubleVar(value=2.0)

        # Valley filtering parameters
        self.valley_prominence_var = tk.DoubleVar(value=1.0)
        self.valley_distance_var = tk.DoubleVar(value=1.0)

        # Clustering
        self.cluster_labels = None
        self.cluster_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        self.show_clusters = False

        self.plotted_cluster_labels = set()
        self.plotted_curve_data = {}

        self.bscan_start_col = 0
        self.bscan_end_col = None

        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        file_frame = ttk.LabelFrame(main_frame, text="File Operations", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(file_frame, text="Open CSV File", command=self.load_csv_file).pack(side=tk.LEFT)
        self.file_path_var = tk.StringVar(value="No file selected")
        ttk.Label(file_frame, textvariable=self.file_path_var).pack(side=tk.LEFT, padx=(10, 0))

        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        bscan_frame = ttk.LabelFrame(display_frame, text="B-Scan Image", padding=5)
        bscan_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.bscan_fig = Figure(figsize=(8, 6), dpi=100)
        self.bscan_ax = self.bscan_fig.add_subplot(111)
        self.bscan_canvas = FigureCanvasTkAgg(self.bscan_fig, bscan_frame)
        self.bscan_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.bscan_canvas.mpl_connect('scroll_event', self.on_bscan_scroll)
        self.bscan_canvas.mpl_connect('button_press_event', self.on_bscan_click)
        NavigationToolbar2Tk(self.bscan_canvas, bscan_frame).update()

        ascan_frame = ttk.LabelFrame(display_frame, text="A-Scan Signal", padding=5)
        ascan_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.ascan_fig = Figure(figsize=(6, 6), dpi=100)
        self.ascan_ax = self.ascan_fig.add_subplot(111)
        self.ascan_canvas = FigureCanvasTkAgg(self.ascan_fig, ascan_frame)
        self.ascan_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.ascan_canvas, ascan_frame).update()

        control_outer_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding=5)
        control_outer_frame.pack(fill=tk.X)

        control_canvas = tk.Canvas(control_outer_frame, height=420)
        control_scrollbar = ttk.Scrollbar(control_outer_frame, orient="vertical", command=control_canvas.yview)
        control_frame = ttk.Frame(control_canvas)
        control_frame.bind("<Configure>", lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all")))
        control_canvas.create_window((0, 0), window=control_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=control_scrollbar.set)
        control_canvas.pack(side="left", fill="both", expand=True)
        control_scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event): control_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _bind_to_mousewheel(event): control_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        def _unbind_from_mousewheel(event): control_canvas.unbind_all("<MouseWheel>")

        control_canvas.bind('<Enter>', _bind_to_mousewheel)
        control_canvas.bind('<Leave>', _unbind_from_mousewheel)

        bscan_ctrl = ttk.Frame(control_frame)
        bscan_ctrl.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(bscan_ctrl, text="B-Scan Range:").pack(side=tk.LEFT)
        ttk.Label(bscan_ctrl, text="Start Trace:").pack(side=tk.LEFT, padx=(20, 5))
        self.start_col_var = tk.StringVar(value="0")
        ttk.Entry(bscan_ctrl, textvariable=self.start_col_var, width=8).pack(side=tk.LEFT)
        ttk.Label(bscan_ctrl, text="End Trace:").pack(side=tk.LEFT, padx=(10, 5))
        self.end_col_var = tk.StringVar(value="")
        ttk.Entry(bscan_ctrl, textvariable=self.end_col_var, width=8).pack(side=tk.LEFT)
        ttk.Button(bscan_ctrl, text="Update", command=self.update_bscan_range).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(bscan_ctrl, text="Reset", command=self.reset_bscan_range).pack(side=tk.LEFT, padx=(5, 0))

        ascan_ctrl = ttk.Frame(control_frame)
        ascan_ctrl.pack(fill=tk.X, pady=(5, 10))
        ttk.Label(ascan_ctrl, text="A-Scan Trace:").pack(side=tk.LEFT)
        self.column_var = tk.StringVar(value="0")
        ttk.Entry(ascan_ctrl, textvariable=self.column_var, width=10).pack(side=tk.LEFT, padx=(5, 5))
        ttk.Button(ascan_ctrl, text="Show", command=self.display_ascan).pack(side=tk.LEFT)

        options_frame = ttk.LabelFrame(control_frame, text="A-Scan Options", padding=5)
        options_frame.pack(fill=tk.X, pady=(0, 10))

        row1 = ttk.Frame(options_frame)
        row1.pack(fill=tk.X, pady=(0, 5))
        ttk.Checkbutton(row1, text="Show Envelope", variable=self.show_envelope_var,
                        command=self.update_ascan_display).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Checkbutton(row1, text="Shared Axis", variable=self.same_plot_var,
                        command=self.update_ascan_display).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Checkbutton(row1, text="Extract Envelope Peaks", variable=self.extract_env_peaks_var,
                        command=self.update_ascan_display).pack(side=tk.LEFT, padx=(0, 20))

        row2 = ttk.Frame(options_frame)
        row2.pack(fill=tk.X)
        ttk.Checkbutton(row2, text="Extract Valleys", variable=self.extract_valleys_var,
                        command=self.update_ascan_display).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Checkbutton(row2, text="Filter Peaks", variable=self.filter_peaks_var,
                        command=self.update_ascan_display).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Checkbutton(row2, text="Match Valleys", variable=self.match_valleys_var,
                        command=self.update_ascan_display).pack(side=tk.LEFT)

        ops_frame = ttk.LabelFrame(control_frame, text="Main Processing Steps", padding=5)
        ops_frame.pack(fill=tk.X, pady=(5, 0))

        id_frame = ttk.LabelFrame(ops_frame, text="Step 1: Initial Boundary Detection", padding=5)
        id_frame.pack(fill=tk.X, pady=5)
        ttk.Button(id_frame, text="Run Detection", command=self.identify_all_boundaries).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(id_frame, text="(Detect boundary candidates in all traces)").pack(side=tk.LEFT, padx=(10, 0))

        pca_filter_frame = ttk.LabelFrame(ops_frame, text="Step 2: PCA Geometric Filtering (Optional)", padding=5)
        pca_filter_frame.pack(fill=tk.X, pady=(5, 5))
        ttk.Label(pca_filter_frame, text="Neighborhood k:").pack(side=tk.LEFT)
        self.pca_k_var = tk.IntVar(value=15)
        ttk.Entry(pca_filter_frame, textvariable=self.pca_k_var, width=8).pack(side=tk.LEFT, padx=(5, 15))
        ttk.Label(pca_filter_frame, text="Aspect Ratio Threshold:").pack(side=tk.LEFT)
        self.pca_ratio_threshold_var = tk.DoubleVar(value=20.0)
        ttk.Entry(pca_filter_frame, textvariable=self.pca_ratio_threshold_var, width=8).pack(side=tk.LEFT, padx=(5, 15))
        ttk.Button(pca_filter_frame, text="Run PCA Filter", command=self.filter_with_pca).pack(side=tk.LEFT, padx=(10, 0))

        scale_frame = ttk.LabelFrame(ops_frame, text="Step 3: Anisotropic Scaling (Optional)", padding=5)
        scale_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(scale_frame, text="Enable Scaling", variable=self.anisotropic_scaling_var).pack(side=tk.LEFT,
                                                                                                       padx=(5, 10))
        ttk.Label(scale_frame, text="Vertical Scale Factor:").pack(side=tk.LEFT, padx=(5, 5))
        ttk.Entry(scale_frame, textvariable=self.anisotropic_factor_var, width=8).pack(side=tk.LEFT)
        ttk.Label(scale_frame, text="(Value > 1, more horizontal preference)").pack(side=tk.LEFT, padx=(10, 0))

        cluster_frame = ttk.LabelFrame(ops_frame, text="Step 4: DBSCAN Clustering", padding=5)
        cluster_frame.pack(fill=tk.X, pady=(5, 0))

        cluster_params = ttk.Frame(cluster_frame)
        cluster_params.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(cluster_params, text="eps:").pack(side=tk.LEFT)
        self.eps_var = tk.DoubleVar(value=40.0)
        ttk.Entry(cluster_params, textvariable=self.eps_var, width=8).pack(side=tk.LEFT, padx=(5, 15))
        ttk.Label(cluster_params, text="min_samples:").pack(side=tk.LEFT)
        self.min_samples_var = tk.IntVar(value=10)
        ttk.Entry(cluster_params, textvariable=self.min_samples_var, width=8).pack(side=tk.LEFT, padx=(5, 15))
        ttk.Button(cluster_params, text="Run DBSCAN", command=self.run_dbscan_clustering).pack(side=tk.LEFT, padx=(10, 0))

        cluster_display = ttk.Frame(cluster_frame)
        cluster_display.pack(fill=tk.X, pady=(5, 0))
        self.show_clusters_var = tk.BooleanVar()
        ttk.Checkbutton(cluster_display, text="Show Clusters", variable=self.show_clusters_var,
                        command=self.toggle_cluster_display).pack(side=tk.LEFT)
        self.cluster_info_var = tk.StringVar(value="Not clustered yet")
        ttk.Label(cluster_display, textvariable=self.cluster_info_var).pack(side=tk.LEFT, padx=(20, 0))

        top_n_frame = ttk.Frame(cluster_frame)
        top_n_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Checkbutton(top_n_frame, text="Show Top N Clusters", variable=self.show_top_n_var,
                        command=self.display_bscan).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(top_n_frame, from_=1, to=100, textvariable=self.top_n_clusters_var, width=5,
                    command=self.display_bscan).pack(side=tk.LEFT)

        boundary_frame = ttk.LabelFrame(control_frame, text="Step 5: Curve Fitting & Valley Extraction", padding=5)
        boundary_frame.pack(fill=tk.X, pady=(5, 0))

        fit_row = ttk.Frame(boundary_frame)
        fit_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(fit_row, text="Cluster Rank (1=largest):").pack(side=tk.LEFT)
        self.cluster_rank_var = tk.IntVar(value=1)
        ttk.Spinbox(fit_row, from_=1, to=50, textvariable=self.cluster_rank_var, width=5).pack(side=tk.LEFT, padx=(5, 10))
        ttk.Label(fit_row, text="Fit Mode:").pack(side=tk.LEFT, padx=(10, 5))
        fit_options = ["Full Width", "Segment Only", "Closed Convex Hull"]
        ttk.OptionMenu(fit_row, self.fit_mode_var, fit_options[1], *fit_options).pack(side=tk.LEFT)
        ttk.Label(fit_row, text="Outlier Removal (σ):").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Entry(fit_row, textvariable=self.outlier_threshold_var, width=8).pack(side=tk.LEFT)

        action_row = ttk.Frame(boundary_frame)
        action_row.pack(fill=tk.X, pady=(5, 5))
        ttk.Button(action_row, text="Plot Curve", command=self.plot_cluster_curve).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(action_row, text="Clear Curves", command=self.clear_plotted_curves).pack(side=tk.LEFT, padx=(0, 10))

        valley_row = ttk.Frame(boundary_frame)
        valley_row.pack(fill=tk.X, pady=(5, 5))
        ttk.Label(valley_row, text="Valley Filtering:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(valley_row, text="Prominence (P) >").pack(side=tk.LEFT)
        ttk.Entry(valley_row, textvariable=self.valley_prominence_var, width=8).pack(side=tk.LEFT, padx=(5, 15))
        ttk.Label(valley_row, text="Min Distance (D) >").pack(side=tk.LEFT)
        ttk.Entry(valley_row, textvariable=self.valley_distance_var, width=8).pack(side=tk.LEFT, padx=(5, 15))
        ttk.Button(valley_row, text="Extract Valleys", command=self.extract_curve_valleys).pack(side=tk.LEFT, padx=(10, 0))

        other_ops_frame = ttk.Frame(control_frame)
        other_ops_frame.pack(fill=tk.X, pady=5)
        ttk.Button(other_ops_frame, text="Clear All Marks", command=self.clear_markers).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(other_ops_frame, text="Save B-Scan Image", command=self.save_bscan_figure).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(other_ops_frame, text="Save A-Scan Image", command=self.save_ascan_figure).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(other_ops_frame, text="Export Cluster Points", command=self.export_results).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(other_ops_frame, text="Export Curves", command=self.export_curves).pack(side=tk.LEFT)
        ttk.Progressbar(other_ops_frame, variable=self.progress_var, maximum=100).pack(side=tk.LEFT, padx=(20, 0),
                                                                                       fill=tk.X, expand=True)

        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, pady=(10, 0))

    # --- Modified: renamed and adjusted to detect valleys ---
    def extract_curve_valleys(self):
        """
        Detect valley points (local minima) in all plotted curves with filtering settings.
        """
        if not self.plotted_curve_data:
            messagebox.showwarning("Warning", "No curve data to analyze. Please plot curves first.")
            return

        try:
            prominence = self.valley_prominence_var.get()
            if prominence <= 0:
                prominence = None
        except tk.TclError:
            prominence = None

        try:
            distance = self.valley_distance_var.get()
            if distance < 1:
                distance = None
        except tk.TclError:
            distance = None

        self.curve_valleys.clear()

        for label, curve_data in self.plotted_curve_data.items():
            if curve_data.get('type') != 'line':
                curve_data['valley_indices'] = []
                continue

            x_coords = curve_data['x']
            y_coords = curve_data['y']

            valley_indices, _ = find_peaks(-y_coords, prominence=prominence, distance=distance)

            if valley_indices.size > 0:
                curve_data['valley_indices'] = valley_indices
                valleys_x = x_coords[valley_indices]
                valleys_y = y_coords[valley_indices]
                self.curve_valleys.extend(zip(valleys_x, valleys_y))
            else:
                curve_data['valley_indices'] = []

        if not self.curve_valleys:
            messagebox.showinfo("Info", "No valleys found under current filtering conditions.")
            self.status_var.set("No valley points detected.")
        else:
            self.status_var.set(f"{len(self.curve_valleys)} valley points extracted.")

        self.display_bscan()

    def save_bscan_figure(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No B-Scan image to save.")
            return
        fp = filedialog.asksaveasfilename(title="Save B-Scan Image", initialfile='b-scan_figure.png',
                                          defaultextension=".png",
                                          filetypes=[("PNG Image", "*.png"), ("TIFF Image", "*.tiff"),
                                                     ("JPEG Image", "*.jpg"), ("PDF Document", "*.pdf"),
                                                     ("All Files", "*.*")])
        if not fp:
            return
        try:
            self.bscan_fig.savefig(fp, bbox_inches='tight', dpi=300)
            messagebox.showinfo("Success", f"B-Scan image saved to:\n{fp}")
            self.status_var.set("B-Scan image saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save B-Scan image:\n{e}")
            self.status_var.set("Save failed.")

    def save_ascan_figure(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No A-Scan image to save.")
            return
        fp = filedialog.asksaveasfilename(title="Save A-Scan Image", initialfile='a-scan_figure.png',
                                          defaultextension=".png",
                                          filetypes=[("PNG Image", "*.png"), ("TIFF Image", "*.tiff"),
                                                     ("JPEG Image", "*.jpg"), ("PDF Document", "*.pdf"),
                                                     ("All Files", "*.*")])
        if not fp:
            return
        try:
            self.ascan_fig.savefig(fp, bbox_inches='tight', dpi=300)
            messagebox.showinfo("Success", f"A-Scan image saved to:\n{fp}")
            self.status_var.set("A-Scan image saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save A-Scan image:\n{e}")
            self.status_var.set("Save failed.")

    def identify_all_boundaries(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        self.clear_markers()
        self.status_var.set("Running initial boundary detection...")
        self.root.update_idletasks()

        bpts = []
        ncol = self.data.shape[1]

        for c in range(ncol):
            if c % 50 == 0:
                self.progress_var.set(c / ncol * 100)
                self.root.update_idletasks()

            sig = self.data.iloc[:, c].values
            env = np.abs(hilbert(sig))
            peaks, _ = find_peaks(env, height=env.max() * 0.1, distance=10)
            fpk = self.filter_envelope_peaks(env, peaks)

            if fpk.size > 0:
                valleys, _ = find_peaks(-sig, height=-sig.min() * 0.1, distance=10)
                if valleys.size > 0:
                    mv = self.match_all_valleys_to_peaks(sig, fpk, valleys)
                    for v in mv:
                        bpts.append((c, v))

        self.original_boundary_points = list(bpts)
        self.boundary_points = list(bpts)
        self.progress_var.set(100)

        self.status_var.set(f"Detection done. {len(bpts)} points found.")
        messagebox.showinfo("Detection Finished",
                            f"Initial boundary detection completed.\n\n"
                            f"Total points found: {len(bpts)}\n\n"
                            f"You may now run PCA filtering or directly run DBSCAN clustering.")

        self.display_bscan()

    def filter_with_pca(self):
        k = self.pca_k_var.get()
        ratio_threshold = self.pca_ratio_threshold_var.get()

        if not self.original_boundary_points or len(self.original_boundary_points) < k + 1:
            messagebox.showwarning("Warning", "Insufficient points for PCA filtering.\nPlease run Step 1 first.")
            self.status_var.set("Too few points for PCA filtering.")
            return

        self.status_var.set(f"Running PCA filtering (k={k})...")
        self.root.update_idletasks()

        points = np.array(self.original_boundary_points)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(points)
        indices = nn.kneighbors(points, return_distance=False)

        good_point_mask = np.zeros(len(points), dtype=bool)
        pca = PCA(n_components=2)

        for i in range(len(points)):
            if i % 200 == 0:
                self.progress_var.set((i / len(points)) * 100)
                self.root.update_idletasks()

            neighborhood_points = points[indices[i]]
            pca.fit(neighborhood_points)
            lambda1, lambda2 = pca.explained_variance_

            if lambda2 < 1e-9:
                is_linear = True
            else:
                is_linear = (lambda1 / lambda2) > ratio_threshold

            if is_linear:
                good_point_mask[i] = True

        num_before = len(self.original_boundary_points)
        self.boundary_points = [p for i, p in enumerate(self.original_boundary_points) if good_point_mask[i]]
        num_after = len(self.boundary_points)

        self.cluster_labels = None
        self.show_clusters_var.set(False)
        self.clear_plotted_curves()
        self.progress_var.set(100)

        self.status_var.set(f"PCA filtering completed. Retained {num_after} / {num_before} points.")
        messagebox.showinfo("PCA Finished",
                            f"PCA filtering completed.\n\n"
                            f"Original points: {num_before}\n"
                            f"Retained points: {num_after}\n\n"
                            f"You may now run DBSCAN clustering.")

        self.display_bscan()

    def run_dbscan_clustering(self):
        if not self.boundary_points:
            messagebox.showwarning("Warning", "No boundary points for clustering.")
            return

        try:
            eps = self.eps_var.get()
            min_samples = self.min_samples_var.get()

            if eps <= 0 or min_samples <= 0:
                messagebox.showerror("Error", "Invalid DBSCAN parameters.")
                return

            points_for_clustering = np.array(self.boundary_points)

            if self.anisotropic_scaling_var.get():
                scaling_factor = self.anisotropic_factor_var.get()
                if scaling_factor <= 0:
                    messagebox.showwarning("Warning", "Scaling factor must be positive.")
                    return

                points_for_clustering = points_for_clustering.astype(float)
                points_for_clustering[:, 1] *= scaling_factor
                self.status_var.set(f"Applying scaling factor {scaling_factor} and running DBSCAN...")
            else:
                self.status_var.set("Running DBSCAN clustering...")

            self.root.update_idletasks()
            self.clear_plotted_curves()

            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            self.cluster_labels = dbscan.fit_predict(points_for_clustering)

            n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
            n_noise = list(self.cluster_labels).count(-1)

            info_text = f"Clusters: {n_clusters}, Noise: {n_noise}"
            self.cluster_info_var.set(info_text)
            self.show_clusters_var.set(True)

            self.display_bscan()
            self.status_var.set(f"DBSCAN completed: {n_clusters} clusters.")

        except Exception as e:
            messagebox.showerror("Error", f"DBSCAN failed:\n{e}")
            self.status_var.set("DBSCAN failed.")

    def display_bscan(self):
        if self.data is None:
            return

        ax = self.bscan_ax
        ax.clear()

        sc = self.bscan_start_col
        ec = self.bscan_end_col or self.data.shape[1]

        ax.imshow(self.data.iloc[:, sc:ec].values, cmap='gray', aspect='auto',
                  extent=[sc, ec, self.data.shape[0], 0])

        if self.plotted_curve_data:
            for label, curve_data in self.plotted_curve_data.items():
                if curve_data['type'] == 'line':
                    ax.plot(curve_data['x'], curve_data['y'], color=curve_data['color'], linewidth=2.5)
                elif curve_data['type'] == 'hull':
                    ax.plot(curve_data['x'], curve_data['y'], color=curve_data['color'], linewidth=2.0)

        pts_to_display = self.boundary_points
        pts = [(x, y) for x, y in pts_to_display if sc <= x < ec]

        if pts and self.show_clusters_var.get() and self.cluster_labels is not None:
            point_to_label_map = {pt: lbl for pt, lbl in zip(self.boundary_points, self.cluster_labels)}
            all_pts_in_range = pts
            all_labels_in_range = [point_to_label_map.get(pt, -1) for pt in all_pts_in_range]

            xs, ys = zip(*all_pts_in_range)

            ranked_labels = self._get_ranked_cluster_labels()
            labels_to_plot = set(ranked_labels)

            if self.show_top_n_var.get():
                top_n = self.top_n_clusters_var.get()
                labels_to_plot &= set(ranked_labels[:top_n])

            rank_map = {label: rank for rank, label in enumerate(ranked_labels)}

            for label in labels_to_plot:
                if label in self.plotted_cluster_labels:
                    continue

                mask = np.array([l == label for l in all_labels_in_range])

                if label in rank_map:
                    rank = rank_map[label]
                    color = self.cluster_colors[rank % len(self.cluster_colors)]
                    ax.scatter(np.array(xs)[mask], np.array(ys)[mask], c=color, s=12, alpha=0.8, marker='o')

        elif pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, c='red', s=8, alpha=0.7, marker='o')

        if self.curve_valleys:
            valley_xs, valley_ys = zip(*self.curve_valleys)
            ax.plot(valley_xs, valley_ys, 'ko', markersize=5)

        ax.set_xlabel('Trace Number')
        ax.set_ylabel('Sample Number')
        self.bscan_canvas.draw()

    def plot_cluster_curve(self):
        if self.cluster_labels is None:
            messagebox.showwarning("Warning", "Please run clustering first.")
            return

        rank_to_plot = self.cluster_rank_var.get() - 1
        ranked_labels = self._get_ranked_cluster_labels()

        if rank_to_plot < 0 or rank_to_plot >= len(ranked_labels):
            messagebox.showwarning("Warning", f"Invalid rank. Total clusters: {len(ranked_labels)}")
            return

        target_label = ranked_labels[rank_to_plot]

        if target_label in self.plotted_cluster_labels:
            messagebox.showinfo("Info", "Curve already plotted.")
            return

        cluster_points = np.array(
            [pt for i, pt in enumerate(self.boundary_points) if self.cluster_labels[i] == target_label]
        )

        fit_mode = self.fit_mode_var.get()
        color = self.cluster_colors[rank_to_plot % len(self.cluster_colors)]

        try:
            outlier_threshold = self.outlier_threshold_var.get()
            if outlier_threshold <= 0:
                messagebox.showwarning("Warning", "Outlier threshold must be positive.")
                return
        except tk.TclError:
            messagebox.showerror("Error", "Invalid outlier threshold.")
            return

        if len(cluster_points) < 4:
            messagebox.showwarning("Warning", "Too few points (<4).")
            return

        x_arr, y_arr = cluster_points[:, 0], cluster_points[:, 1]
        med = np.median(y_arr)
        dev = np.std(np.abs(y_arr - med))

        if dev == 0:
            clean_points = cluster_points
        else:
            mask = np.abs(y_arr - med) < outlier_threshold * dev
            clean_points = cluster_points[mask]

        if len(clean_points) < 3:
            messagebox.showwarning("Warning", "Too few points after outlier removal.")
            return

        if fit_mode == "Closed Convex Hull":
            try:
                hull = ConvexHull(clean_points)
                hull_vertices = clean_points[hull.vertices]

                distances = np.sqrt(np.sum(np.diff(hull_vertices, axis=0)**2, axis=1))
                cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

                points_to_interp = np.vstack([hull_vertices, hull_vertices[0]])
                param_t = np.append(
                    cumulative_distances,
                    cumulative_distances[-1] + np.linalg.norm(hull_vertices[-1] - hull_vertices[0])
                )

                if len(param_t) < 4:
                    self.plotted_curve_data[target_label] = {
                        'x': points_to_interp[:, 0], 'y': points_to_interp[:, 1],
                        'color': color, 'type': 'hull'
                    }
                else:
                    t_smooth = np.linspace(param_t.min(), param_t.max(), 300)
                    fx = interp1d(param_t, points_to_interp[:, 0], kind='cubic')
                    fy = interp1d(param_t, points_to_interp[:, 1], kind='cubic')
                    x_smooth = fx(t_smooth)
                    y_smooth = fy(t_smooth)
                    self.plotted_curve_data[target_label] = {
                        'x': x_smooth, 'y': y_smooth, 'color': color, 'type': 'hull'
                    }

            except Exception as e:
                messagebox.showerror("Error", f"Convex hull error:\n{e}")
                return

        else:
            if len(clean_points) < 4:
                messagebox.showwarning("Warning", "Spline interpolation requires ≥4 points.")
                return

            x_good, y_good = clean_points[:, 0], clean_points[:, 1]
            sort_indices = np.argsort(x_good)
            x_good, y_good = x_good[sort_indices], y_good[sort_indices]

            if fit_mode == "Segment Only":
                if x_good.max() == x_good.min():
                    messagebox.showwarning("Warning", "Zero-width segment.")
                    return
                cols = np.linspace(x_good.min(), x_good.max(),
                                   num=int(x_good.max() - x_good.min()) * 2 + 1)
            else:
                cols = np.arange(self.data.shape[1])

            try:
                if len(np.unique(x_good)) < 2:
                    raise ValueError("Interpolation requires at least two unique x values.")

                f_spline = interp1d(x_good, y_good, kind='cubic',
                                    bounds_error=False, fill_value="extrapolate")
                y_interp = f_spline(cols)

            except Exception as e:
                messagebox.showerror("Error",
                                     f"Spline failed:\n{e}\n\nFalling back to linear interpolation.")
                y_interp = np.interp(cols, x_good, y_good)

            self.plotted_curve_data[target_label] = {
                'x': cols, 'y': y_interp, 'color': color, 'type': 'line'
            }

        self.plotted_cluster_labels.add(target_label)
        self.status_var.set(f"Plotted curve for cluster rank {rank_to_plot + 1}")
        self.display_bscan()

    def update_ascan_display(self):
        if self.data is None:
            return

        ax = self.ascan_ax
        ax.clear()

        sig = self.data.iloc[:, self.current_column].values
        x = np.arange(len(sig))
        ax.plot(x, sig, 'b-', label='Signal')

        if self.show_envelope_var.get():
            env = np.abs(hilbert(sig))
            if self.same_plot_var.get():
                ax.plot(x, env, 'r-', label='Envelope')
            else:
                ax2 = ax.twinx()
                ax2.plot(x, env, 'r-', label='Envelope')
                ax2.set_ylabel('Amplitude', color='r')

        if self.extract_env_peaks_var.get():
            env = np.abs(hilbert(sig))
            peaks, _ = find_peaks(env, height=env.max() * 0.1, distance=10)
            self.envelope_peaks = peaks
            ax.plot(peaks, env[peaks], 'ro', label='Envelope Peaks')

            if self.filter_peaks_var.get():
                filt = self.filter_envelope_peaks(env, peaks)
                self.filtered_peaks = filt
                ax.plot(filt, env[filt], 'go', label='Filtered Peaks')

        if self.extract_valleys_var.get():
            valleys, _ = find_peaks(-sig, height=-sig.min() * 0.1, distance=10)
            self.signal_valleys = valleys
            ax.plot(valleys, sig[valleys], 'co', label='Valleys')

            if self.match_valleys_var.get() and hasattr(self, 'filtered_peaks') and self.filtered_peaks.size > 0:
                mv = self.match_all_valleys_to_peaks(sig, self.filtered_peaks, valleys)
                if mv.size > 0:
                    ax.plot(mv, sig[mv], 'mo', label='Matched Valleys')

        ax.set_xlabel('Sample Number')
        ax.set_ylabel('Amplitude')
        ax.grid(False)
        self.ascan_canvas.draw()

    def _get_ranked_cluster_labels(self):
        if self.cluster_labels is None:
            return []

        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        valid_labels_mask = unique_labels != -1
        unique_labels = unique_labels[valid_labels_mask]
        counts = counts[valid_labels_mask]

        if len(counts) == 0:
            return []

        sorted_indices = np.argsort(-counts)
        return unique_labels[sorted_indices]

    def clear_plotted_curves(self):
        if not self.plotted_curve_data and not self.curve_valleys:
            return

        self.plotted_curve_data.clear()
        self.plotted_cluster_labels.clear()
        self.curve_valleys.clear()

        self.status_var.set("Cleared all curves and valley points.")
        self.display_bscan()

    def clear_markers(self):
        self.original_boundary_points = []
        self.boundary_points = []
        self.cluster_labels = None
        self.show_clusters_var.set(False)
        self.show_top_n_var.set(False)
        self.cluster_info_var.set("Not clustered yet")
        self.progress_var.set(0)

        self.clear_plotted_curves()
        if self.data is not None:
            self.display_bscan()
            self.update_ascan_display()

        self.status_var.set("Cleared all markers and curves")

    def load_csv_file(self):
        fp = filedialog.askopenfilename(title="Open CSV File",
                                        filetypes=[("CSV", "*.csv"), ("All Files", "*.*")])
        if not fp:
            return

        self.clear_markers()

        try:
            self.status_var.set("Loading file...")
            self.root.update_idletasks()

            self.data = pd.read_csv(fp, header=None)
            self.file_path_var.set(os.path.basename(fp))

            self.bscan_start_col, self.bscan_end_col = 0, self.data.shape[1]
            self.start_col_var.set("0")
            self.end_col_var.set(str(self.bscan_end_col))

            self.display_bscan()

            self.current_column = 0
            self.column_var.set("0")
            self.display_ascan()

            self.status_var.set(f"Loaded: {self.data.shape[0]} × {self.data.shape[1]}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            self.status_var.set("Load failed")

    def export_curves(self):
        if not self.plotted_curve_data:
            messagebox.showwarning("Warning", "No curves to export. Please plot curves first.")
            return

        fp = filedialog.asksaveasfilename(title="Save Curves", defaultextension=".csv",
                                          filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not fp:
            return

        try:
            export_dfs = []
            ranked_labels = self._get_ranked_cluster_labels()
            rank_map = {label: rank + 1 for rank, label in enumerate(ranked_labels)}

            for label, curve in self.plotted_curve_data.items():
                rank = rank_map.get(label, "N/A")
                df = pd.DataFrame({'x': curve['x'], 'y': curve['y']})

                df['is_valley'] = False
                if 'valley_indices' in curve and len(curve['valley_indices']) > 0:
                    df.loc[curve['valley_indices'], 'is_valley'] = True

                df['curve_type'] = curve['type']
                df['cluster_rank'] = rank
                export_dfs.append(df)

            final_df = pd.concat(export_dfs, ignore_index=True)
            final_df = final_df[['cluster_rank', 'curve_type', 'x', 'y', 'is_valley']]
            final_df.to_csv(fp, index=False, encoding='utf-8-sig')

            self.status_var.set(f"Curves saved to {os.path.basename(fp)}")
            messagebox.showinfo("Success", f"Curves exported to:\n{fp}")

        except Exception as e:
            self.status_var.set("Export failed.")
            messagebox.showerror("Error", f"Failed to export curves:\n{e}")

    def update_bscan_range(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        try:
            sc = int(self.start_col_var.get())
            ec = int(self.end_col_var.get() or self.data.shape[1])

            if sc < 0 or ec <= sc or ec > self.data.shape[1]:
                raise ValueError

            self.bscan_start_col, self.bscan_end_col = sc, ec
            self.display_bscan()
            self.status_var.set(f"B-Scan range: {sc} – {ec - 1}")

        except:
            messagebox.showerror("Error", "Invalid trace range")

    def reset_bscan_range(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        self.bscan_start_col, self.bscan_end_col = 0, self.data.shape[1]
        self.start_col_var.set("0")
        self.end_col_var.set(str(self.data.shape[1]))

        self.display_bscan()
        self.status_var.set("Range reset")

    def display_ascan(self):
        if self.data is None:
            return

        try:
            col = int(self.column_var.get())
            if col < 0 or col >= self.data.shape[1]:
                raise ValueError

            self.current_column = col
            self.update_ascan_display()

        except:
            messagebox.showerror("Error", "Invalid trace number")

    def filter_envelope_peaks(self, envelope, peaks):
        if len(peaks) == 0:
            return np.array([])
        filtered = []
        for i, p in enumerate(peaks):
            if i == 0:
                if len(peaks) > 1 and envelope[p] > envelope[peaks[i + 1]]:
                    filtered.append(p)
                elif len(peaks) == 1:
                    filtered.append(p)
            elif i == len(peaks) - 1:
                continue
            else:
                if envelope[p] > envelope[peaks[i - 1]] and envelope[p] > envelope[peaks[i + 1]]:
                    filtered.append(p)
        return np.array(filtered)

    def match_all_valleys_to_peaks(self, signal, peaks, valleys):
        if len(peaks) == 0 or len(valleys) == 0:
            return np.array([])
        matched = []
        for p in peaks:
            idx = np.argmin(np.abs(valleys - p))
            matched.append(valleys[idx])
        return np.array(list(set(matched)))

    def toggle_cluster_display(self):
        if self.cluster_labels is None and self.show_clusters_var.get():
            messagebox.showwarning("Warning", "No clustering result available.")
            self.show_clusters_var.set(False)
            return
        self.display_bscan()
        self.status_var.set("Toggled cluster display")

    def on_bscan_scroll(self, event):
        if event.inaxes != self.bscan_ax:
            return
        xlim = self.bscan_ax.get_xlim()
        ylim = self.bscan_ax.get_ylim()
        zoom_factor = 1.1 if event.step < 0 else 1 / 1.1
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        x_range = (xlim[1] - xlim[0]) * zoom_factor
        y_range = (ylim[1] - ylim[0]) * zoom_factor
        new_xlim = [xdata - x_range / 2, xdata + x_range / 2]
        new_ylim = [ydata - y_range / 2, ydata + y_range / 2]
        self.bscan_ax.set_xlim(new_xlim)
        self.bscan_ax.set_ylim(new_ylim)
        self.bscan_canvas.draw()

    def on_bscan_click(self, event):
        if event.inaxes != self.bscan_ax or event.xdata is None:
            return
        col = int(round(event.xdata))
        if self.data is not None and 0 <= col < self.data.shape[1]:
            self.column_var.set(str(col))
            self.display_ascan()

    def export_results(self):
        if self.cluster_labels is None:
            messagebox.showwarning("Warning", "No clustering results to export.")
            return

        fp = filedialog.asksaveasfilename(defaultextension=".csv",
                                          filetypes=[("CSV", "*.csv")])
        if not fp:
            return

        points_to_export = self.boundary_points
        labels_to_export = self.cluster_labels

        if len(points_to_export) != len(labels_to_export):
            messagebox.showerror("Error", "Mismatch between points and labels.")
            return

        export_data = []
        for i, (x, y) in enumerate(points_to_export):
            row = {'Trace': x, 'Position': y, 'Cluster': labels_to_export[i]}
            export_data.append(row)

        df = pd.DataFrame(export_data)
        df.to_csv(fp, index=False, encoding='utf-8-sig')

        messagebox.showinfo("Success", f"Data saved to {fp}")
        self.status_var.set("Export succeeded")


def main():
    root = tk.Tk()
    app = GPRAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
