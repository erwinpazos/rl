import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import xml.etree.ElementTree as ET

# --- CORRIDOR PARAMETERS ---
CELL_SIZE = 0.5  # Cell size (side) in meters: 0.5m
CORRIDOR_LENGTH_M = 100.0
CORRIDOR_WIDTH_M = 3.0

NUM_CELLS_X = int(CORRIDOR_LENGTH_M / CELL_SIZE) # 200 (Length)
NUM_CELLS_Y = int(CORRIDOR_WIDTH_M / CELL_SIZE) # 6 (Width)

# --- TKINTER / MUJOCO COLORS CONFIGURATION ---
CANVAS_CELL_PX = 40 # Enlarged cell size
RULER_WIDTH_PX = 70 # Ruler area width
CANVAS_BG_COLOR = "#ffffff"
GRID_LINE_COLOR = "#e0e0e0"

# --- MODERN COLOR THEME ---
COLORS = {
  'primary': '#2563eb',
  'primary_hover': '#1d4ed8',
  'secondary': '#64748b', 
  'success': '#10b981',
  'danger': '#ef4444',
  'warning': '#f59e0b',
  'info': '#06b6d4',
  'light': '#f8fafc',
  'dark': '#1e293b',
  'white': '#ffffff',
  'border': '#e2e8f0',
  'hover': '#f1f5f9',
  'ruler_bg': '#f1f5f9',
  'ruler_text': '#475569'
}

# Heights definition (Half-size in MuJoCo) and colors
BUMP_SETTINGS = {
  "Small (0.05m)": {"half_z": 0.025, "tk_color": "#fbbf24", "mujoco_rgb": "1 1 0.2", "mat_name": "mat_bump_small"},
  "Medium (0.2m)": {"half_z": 0.1, "tk_color": "#f97316", "mujoco_rgb": "1 0.5 0", "mat_name": "mat_bump_medium"},
  "Large (0.5m)": {"half_z": 0.25, "tk_color": "#dc2626", "mujoco_rgb": "0.8 0 0", "mat_name": "mat_bump_large"}
}
BUMP_KEYS = list(BUMP_SETTINGS.keys())
BASE_FLAT_HALF_Z = 0.025 

class CorridorEditor:
    def __init__(self, master):
        self.master = master
        master.title("MuJoCo Corridor Editor")
        master.geometry("1400x900")
        master.configure(bg=COLORS['light'])
        
        self.setup_styles()

        self.current_tool = "flat"
        self.drawing = False
        
        self.grid = [['flat' for _ in range(NUM_CELLS_Y)] for _ in range(NUM_CELLS_X)]
        
        self.bump_height_var = tk.StringVar(master)
        self.bump_height_var.set(BUMP_KEYS[0])

        self.create_main_interface()
        
    def setup_styles(self):
        """Configure ttk styles for a modern appearance"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Primary button style
        style.configure('Primary.TButton',
                        background=COLORS['primary'],
                        foreground='white',
                        borderwidth=0,
                        focuscolor='none',
                        padding=(20, 10),
                        font=('Segoe UI', 10))
        
        style.map('Primary.TButton',
                  background=[('active', COLORS['primary_hover']),
                             ('pressed', COLORS['primary_hover'])],
                  foreground=[('active', 'white'),
                             ('pressed', 'white')])
        
        # Tool button style
        style.configure('Tool.TButton',
                        background=COLORS['white'],
                        foreground=COLORS['dark'],
                        borderwidth=1,
                        relief='solid',
                        focuscolor='none',
                        padding=(16, 8),
                        font=('Segoe UI', 10))
        
        style.map('Tool.TButton',
                  background=[('active', COLORS['hover']),
                              ('pressed', COLORS['primary'])],
                  foreground=[('active', COLORS['dark']),
                             ('pressed', 'white')],
                  bordercolor=[('active', COLORS['primary'])])
        
        # Selected button style
        style.configure('Selected.TButton',
                        background=COLORS['primary'],
                        foreground='white',
                        borderwidth=2,
                        relief='solid',
                        focuscolor='none',
                        padding=(16, 8),
                        font=('Segoe UI', 10, 'bold'))
        
        style.map('Selected.TButton',
                  background=[('active', COLORS['primary_hover'])],
                  foreground=[('active', 'white')])
        
        # Combobox style
        style.configure('Modern.TCombobox',
                        fieldbackground=COLORS['white'],
                        background=COLORS['white'],
                        borderwidth=1,
                        relief='solid',
                        arrowcolor=COLORS['primary'],
                        padding=5,
                        font=('Segoe UI', 10))
        
        style.map('Modern.TCombobox',
                  fieldbackground=[('readonly', COLORS['white'])],
                  selectbackground=[('readonly', COLORS['white'])],
                  selectforeground=[('readonly', COLORS['dark'])])
        
        style.configure('TFrame', background=COLORS['light'])
        style.configure('TLabel', background=COLORS['light'], font=('Segoe UI', 10))
        style.configure('TLabelframe', 
                       background=COLORS['white'], 
                       borderwidth=1, 
                       relief='solid',
                       bordercolor=COLORS['border'])
        style.configure('TLabelframe.Label', 
                       background=COLORS['white'], 
                       foreground=COLORS['dark'],
                       font=('Segoe UI', 10, 'bold'))

        
    def create_main_interface(self):
        """Create main interface with modern design"""
        self.main_frame = ttk.Frame(self.master, style='TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.create_header()
        self.create_toolbar()
        self.create_canvas_area()
        self.create_status_bar()
        
    def create_header(self):
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(header_frame, 
                                text="MuJoCo Corridor Editor",
                                font=('Segoe UI', 24, 'bold'),
                                foreground=COLORS['dark'])
        title_label.pack(side=tk.LEFT)
        
        info_frame = ttk.Frame(header_frame)
        info_frame.pack(side=tk.RIGHT)
        
        info_text = f"Dimensions: {CORRIDOR_LENGTH_M}m × {CORRIDOR_WIDTH_M}m | Cells: {NUM_CELLS_X} × {NUM_CELLS_Y} | Cell size: {CELL_SIZE}m"
        info_label = ttk.Label(info_frame,
                                text=info_text,
                                font=('Segoe UI', 9),
                                foreground=COLORS['secondary'])
        info_label.pack()
        
    def create_toolbar(self):
        self.toolbar = ttk.Frame(self.main_frame)
        self.toolbar.pack(fill=tk.X, pady=(0, 20))
        
        # File section
        file_frame = ttk.LabelFrame(self.toolbar, text="File", padding=15)
        file_frame.pack(side=tk.LEFT, padx=(0, 15), fill=tk.Y)
        
        self.import_button = ttk.Button(file_frame, 
                                        text="Import XML", 
                                        command=self.import_xml,
                                        style='Primary.TButton')
        self.import_button.pack(side=tk.LEFT, padx=(0, 8))
        
        self.generate_button = ttk.Button(file_frame, 
                                            text="Generate XML", 
                                            command=self.generate_xml,
                                            style='Primary.TButton')
        self.generate_button.pack(side=tk.LEFT)
        
        # Tools section
        tools_frame = ttk.LabelFrame(self.toolbar, text="Drawing Tools", padding=15)
        tools_frame.pack(side=tk.LEFT, padx=(0, 15), fill=tk.Y)
        
        self.flat_button = ttk.Button(tools_frame, 
                                        text="Flat Floor", 
                                        command=lambda: self.set_tool("flat"),
                                        style='Tool.TButton',
                                        width=12)
        self.flat_button.pack(side=tk.LEFT, padx=(0, 8))
        
        self.bump_button = ttk.Button(tools_frame, 
                                        text="Bump", 
                                        command=lambda: self.set_tool("bump"),
                                        style='Tool.TButton',
                                        width=12)
        self.bump_button.pack(side=tk.LEFT, padx=(0, 8))
        
        self.hole_button = ttk.Button(tools_frame, 
                                        text="Erase", 
                                        command=lambda: self.set_tool("hole"),
                                        style='Tool.TButton',
                                        width=12)
        self.hole_button.pack(side=tk.LEFT)
        
        # Bump configuration section
        bump_config_frame = ttk.LabelFrame(self.toolbar, text="Bump Configuration", padding=15)
        bump_config_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(bump_config_frame, text="Height:", font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(0, 8))
        
        self.bump_menu = ttk.Combobox(bump_config_frame, 
                                        textvariable=self.bump_height_var,
                                        values=BUMP_KEYS,
                                        state="readonly",
                                        width=18,
                                        style='Modern.TCombobox')
        self.bump_menu.pack(side=tk.LEFT)
        
        self.set_tool("flat")

    def create_canvas_area(self):
        """Create drawing area with ruler and centering."""
        
        canvas_container = ttk.Frame(self.main_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        canvas_title = ttk.Label(canvas_container, 
                                 text="Drawing Area",
                                 font=('Segoe UI', 14, 'bold'),
                                 foreground=COLORS['dark'])
        canvas_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Main centered frame
        center_frame = ttk.Frame(canvas_container)
        center_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame for grid (Legend + Ruler + Canvas)
        grid_frame = ttk.Frame(center_frame)
        grid_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Legend on the left
        self.create_legend(grid_frame)

        # Ruler canvas
        canvas_height_px = NUM_CELLS_X * CANVAS_CELL_PX
        
        self.ruler_canvas = tk.Canvas(
            grid_frame,
            width=RULER_WIDTH_PX,
            height=min(canvas_height_px, 650),
            bg=COLORS['ruler_bg'],
            highlightthickness=1,
            highlightbackground=COLORS['border'],
            scrollregion=(0, 0, RULER_WIDTH_PX, canvas_height_px)
        )
        self.ruler_canvas.pack(side=tk.LEFT, fill=tk.Y)
        
        # Frame for drawing canvas and vertical scrollbar
        canvas_scroll_frame = ttk.Frame(grid_frame)
        canvas_scroll_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.v_scrollbar = ttk.Scrollbar(canvas_scroll_frame, orient=tk.VERTICAL)
        
        canvas_width_px = NUM_CELLS_Y * CANVAS_CELL_PX
        
        self.canvas = tk.Canvas(
            canvas_scroll_frame,
            width=canvas_width_px,
            height=min(canvas_height_px, 650),
            scrollregion=(0, 0, canvas_width_px, canvas_height_px),
            yscrollcommand=self.v_scrollbar.set,
            bg=CANVAS_BG_COLOR,
            highlightthickness=1,
            highlightbackground=COLORS['border']
        )
        
        # Scrollbar synchronization
        self.v_scrollbar.config(command=lambda *args: (self.canvas.yview(*args), self.ruler_canvas.yview(*args)))
        
        # Placement
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bindings
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.drag_draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)
        self.canvas.bind("<Button-3>", self.show_context_menu)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel_mac)
        self.canvas.bind("<Button-5>", self.on_mousewheel_mac)
        
        self.create_context_menu()
        self.draw_ruler()
        self.draw_grid_lines()
        self.update_canvas_content()
    
    def create_legend(self, parent):
        """Create legend for cell types"""
        legend_frame = ttk.LabelFrame(parent, text="Legend", padding=15)
        legend_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        legend_items = [
            ("Hole", CANVAS_BG_COLOR, COLORS['border']),
            ("Flat Floor", '#10b981', ''),
            ("Small Bump", '#fbbf24', 'white'),
            ("Medium Bump", '#f97316', 'white'),
            ("Large Bump", '#dc2626', 'white')
        ]
        
        for label, fill_color, outline_color in legend_items:
            item_frame = ttk.Frame(legend_frame)
            item_frame.pack(fill=tk.X, pady=5)
            
            # Canvas for color square
            color_canvas = tk.Canvas(item_frame, width=30, height=30, 
                                     bg=COLORS['white'], highlightthickness=0)
            color_canvas.pack(side=tk.LEFT, padx=(0, 10))
            
            if label == "Hole":
                # Empty square with border
                color_canvas.create_rectangle(2, 2, 28, 28, 
                                             fill=fill_color, 
                                             outline=outline_color,
                                             width=2)
            elif "Bump" in label:
                # Square with depth effect
                color_canvas.create_rectangle(2, 2, 28, 28, 
                                             fill=fill_color, 
                                             outline='')
                color_canvas.create_rectangle(6, 6, 24, 24,
                                             fill=fill_color,
                                             outline=outline_color,
                                             width=2)
            else:
                # Simple flat floor
                color_canvas.create_rectangle(2, 2, 28, 28, 
                                             fill=fill_color, 
                                             outline='')
            
            # Text label
            text_label = ttk.Label(item_frame, 
                                  text=label,
                                  font=('Segoe UI', 10),
                                  foreground=COLORS['dark'])
            text_label.pack(side=tk.LEFT)

    def draw_ruler(self):
        """Draw marks and labels on the side ruler."""
        self.ruler_canvas.delete("all")
        
        CELLS_PER_MARK = 10 
        total_height = NUM_CELLS_X * CANVAS_CELL_PX
        
        for i in range(NUM_CELLS_X + 1):
            y_pos = i * CANVAS_CELL_PX
            
            if i % CELLS_PER_MARK == 0:
                # Long mark
                mark_length = 20
                meter_label = i * CELL_SIZE
                
                self.ruler_canvas.create_line(RULER_WIDTH_PX - mark_length, y_pos, RULER_WIDTH_PX, y_pos, 
                                              fill=COLORS['dark'], width=2, tags="ruler_mark")
                
                # Label with offset to avoid overlap
                text_y = y_pos - 3 if y_pos > 0 else y_pos + 3
                self.ruler_canvas.create_text(RULER_WIDTH_PX - mark_length - 8, text_y, 
                                              anchor=tk.E, 
                                              text=f"{meter_label:.0f}m", 
                                              fill=COLORS['ruler_text'], 
                                              font=('Segoe UI', 9, 'bold'))
            elif i % (CELLS_PER_MARK / 2) == 0:
                # Medium mark
                mark_length = 12
                self.ruler_canvas.create_line(RULER_WIDTH_PX - mark_length, y_pos, RULER_WIDTH_PX, y_pos, 
                                              fill=COLORS['secondary'], width=1.5, tags="ruler_mark")
            else:
                # Small mark
                mark_length = 6
                self.ruler_canvas.create_line(RULER_WIDTH_PX - mark_length, y_pos, RULER_WIDTH_PX, y_pos, 
                                              fill=GRID_LINE_COLOR, width=1, tags="ruler_mark")

        # Axis title
        self.ruler_canvas.create_text(RULER_WIDTH_PX / 2, 15, 
                                      anchor=tk.N, 
                                      text="X Axis", 
                                      fill=COLORS['secondary'], 
                                      font=('Segoe UI', 9, 'bold'))
        
    def create_status_bar(self):
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Separator(self.status_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 0))
        
        status_content = ttk.Frame(self.status_frame)
        status_content.pack(fill=tk.X, pady=10)
        
        self.tool_status = ttk.Label(status_content, 
                                        text="Current tool: Flat Floor",
                                        font=('Segoe UI', 10),
                                        foreground=COLORS['dark'])
        self.tool_status.pack(side=tk.LEFT)
        
        self.mouse_status = ttk.Label(status_content,
                                        text="Position: -",
                                        font=('Segoe UI', 9),
                                        foreground=COLORS['secondary'])
        self.mouse_status.pack(side=tk.RIGHT, padx=(10, 0))
        
        shortcuts_label = ttk.Label(status_content,
                                        text="Shortcuts: Ctrl+O (Import) | Ctrl+S (Generate) | 1, 2, 3 (Tools)",
                                        font=('Segoe UI', 9),
                                        foreground=COLORS['info'])
        shortcuts_label.pack(side=tk.RIGHT, padx=(20, 10))
        
        self.canvas.bind("<Motion>", self.update_mouse_position)
        
        self.master.bind('<Control-o>', lambda e: self.import_xml())
        self.master.bind('<Control-s>', lambda e: self.generate_xml())
        self.master.bind('<Key-1>', lambda e: self.set_tool("flat"))
        self.master.bind('<Key-2>', lambda e: self.set_tool("bump"))
        self.master.bind('<Key-3>', lambda e: self.set_tool("hole"))
        self.master.focus_set()
        
    def create_context_menu(self):
        self.context_menu = tk.Menu(self.master, tearoff=0, font=('Segoe UI', 10))
        self.context_menu.add_command(label="Flat Floor", command=lambda: self.set_tool("flat"))
        self.context_menu.add_command(label="Bump", command=lambda: self.set_tool("bump"))
        self.context_menu.add_command(label="Erase", command=lambda: self.set_tool("hole"))
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Refresh", command=self.update_canvas_content)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Import XML", command=self.import_xml)
        self.context_menu.add_command(label="Generate XML", command=self.generate_xml)
        
    def show_context_menu(self, event):
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()
            
    def update_mouse_position(self, event):
        x_mujoco_idx, y_mujoco_idx = self.get_cell_indices(event)
        
        if x_mujoco_idx is not None and y_mujoco_idx is not None:
            mujoco_x_pos = x_mujoco_idx * CELL_SIZE
            mujoco_y_pos_center = (y_mujoco_idx * CELL_SIZE) - (CORRIDOR_WIDTH_M / 2)
            
            self.mouse_status.config(text=f"Position: X={mujoco_x_pos:.1f}m (Cell {x_mujoco_idx}) | Y={mujoco_y_pos_center:.1f}m (Cell {y_mujoco_idx})")
        else:
            self.mouse_status.config(text="Position: Outside grid")
    
    def set_tool(self, tool_name):
        self.current_tool = tool_name
        
        self.flat_button.config(style='Tool.TButton')
        self.bump_button.config(style='Tool.TButton')
        self.hole_button.config(style='Tool.TButton')
        
        tool_names = {
            "flat": ("Flat Floor", self.flat_button),
            "bump": ("Bump", self.bump_button), 
            "hole": ("Erase", self.hole_button)
        }
        
        if tool_name in tool_names:
            _, button = tool_names[tool_name]
            button.config(style='Selected.TButton')
            
        status_text = {
            "flat": "Current tool: Flat Floor - Click to create floor",
            "bump": f"Current tool: Bump ({self.bump_height_var.get()}) - Click to create bumps",
            "hole": "Current tool: Erase - Click to remove elements"
        }
        
        if hasattr(self, 'tool_status'):
            self.tool_status.config(text=status_text.get(tool_name, "Unknown tool"))
            
    def get_cell_indices(self, event):
        x_canvas_coord = self.canvas.canvasx(event.x)
        y_canvas_coord = self.canvas.canvasy(event.y)
        
        y_mujoco_idx = int(x_canvas_coord // CANVAS_CELL_PX)
        x_mujoco_idx = int(y_canvas_coord // CANVAS_CELL_PX)
        
        if 0 <= x_mujoco_idx < NUM_CELLS_X and 0 <= y_mujoco_idx < NUM_CELLS_Y:
            return x_mujoco_idx, y_mujoco_idx
        return None, None

    def start_draw(self, event):
        x, y = self.get_cell_indices(event)
        if x is not None:
            self.drawing = True
            self.apply_tool(x, y)

    def drag_draw(self, event):
        if self.drawing:
            x, y = self.get_cell_indices(event)
            if x is not None:
                self.apply_tool(x, y)

    def end_draw(self, event):
        self.drawing = False

    def apply_tool(self, x_idx, y_idx):
        new_cell_value = self.current_tool
        
        if self.current_tool == 'bump':
            new_cell_value = (self.current_tool, self.bump_height_var.get())
        
        if self.grid[x_idx][y_idx] != new_cell_value:
            self.grid[x_idx][y_idx] = new_cell_value
            self.draw_cell_content(x_idx, y_idx)

    def draw_grid_lines(self):
        """Draw grid lines with modern style"""
        self.canvas.delete("grid_line") 
        total_width = NUM_CELLS_Y * CANVAS_CELL_PX
        total_height = NUM_CELLS_X * CANVAS_CELL_PX

        # Horizontal lines
        for i in range(NUM_CELLS_X + 1):
            y = i * CANVAS_CELL_PX
            width = 2 if i % 10 == 0 else 1
            color = COLORS['secondary'] if i % 10 == 0 else GRID_LINE_COLOR
            self.canvas.create_line(0, y, total_width, y, 
                                    fill=color, width=width, tags="grid_line")

        # Vertical lines 
        for j in range(NUM_CELLS_Y + 1):
            x = j * CANVAS_CELL_PX
            width = 2 if j == NUM_CELLS_Y / 2 else 1
            color = COLORS['primary'] if j == NUM_CELLS_Y / 2 else GRID_LINE_COLOR
            self.canvas.create_line(x, 0, x, total_height, 
                                    fill=color, width=width, tags="grid_line")
        
        self.canvas.tag_raise("grid_line")

    def get_draw_params(self, cell_value):
        if isinstance(cell_value, str):
            cell_type = cell_value
        elif isinstance(cell_value, tuple):
            cell_type, height_key = cell_value
        else:
            return None

        if cell_type == 'flat':
            return '#10b981'
        
        if cell_type == 'bump':
            modern_colors = {
                "Small (0.05m)": "#fbbf24", 
                "Medium (0.2m)": "#f97316", 
                "Large (0.5m)": "#dc2626" 
            }
            return modern_colors.get(height_key, modern_colors[BUMP_KEYS[0]])
            
        return None

    def draw_cell_content(self, x_idx, y_idx):
        cell_value = self.grid[x_idx][y_idx]
        
        self.canvas.delete(f"cell_{x_idx}_{y_idx}")
        
        cell_type = cell_value[0] if isinstance(cell_value, tuple) else cell_value

        if cell_type == 'hole':
            return

        color = self.get_draw_params(cell_value)
        if color:
            x1_canvas = y_idx * CANVAS_CELL_PX
            y1_canvas = x_idx * CANVAS_CELL_PX
            x2_canvas = x1_canvas + CANVAS_CELL_PX
            y2_canvas = y1_canvas + CANVAS_CELL_PX

            # Main rectangle
            self.canvas.create_rectangle(x1_canvas + 1, y1_canvas + 1, x2_canvas - 1, y2_canvas - 1, 
                                         fill=color, 
                                         outline='',
                                         tags=(f"cell_{x_idx}_{y_idx}", "cell_content"))
            
            # Depth effect for bumps
            if cell_type == 'bump':
                margin = 3
                self.canvas.create_rectangle(x1_canvas + margin, y1_canvas + margin, 
                                             x2_canvas - margin, y2_canvas - margin,
                                             fill=color,
                                             outline='white',
                                             width=2,
                                             tags=(f"cell_{x_idx}_{y_idx}", "cell_content"))

        self.canvas.tag_lower("cell_content", "grid_line")

    def update_canvas_content(self):
        self.canvas.delete("cell_content")
        for x in range(NUM_CELLS_X):
            for y in range(NUM_CELLS_Y):
                self.draw_cell_content(x, y)
        self.canvas.tag_raise("grid_line")
        
    def on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.ruler_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def on_mousewheel_mac(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
            self.ruler_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
            self.ruler_canvas.yview_scroll(1, "units")

    def import_xml(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("MuJoCo XML files", "*.xml")],
            title="Select MuJoCo XML file to import"
        )
        
        if not filepath:
            return

        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            self.grid = [['hole' for _ in range(NUM_CELLS_Y)] for _ in range(NUM_CELLS_X)]
            
            y_offset_mujoco = -CORRIDOR_WIDTH_M / 2
            HALF_SIZE_GRID_XY = CELL_SIZE / 2.0 

            bump_half_z_to_key = {settings["half_z"]: key for key, settings in BUMP_SETTINGS.items()}

            for geom in root.findall('./worldbody/geom'):
                name = geom.get('name', '')
                pos_str = geom.get('pos')
                size_str = geom.get('size')
                
                if not pos_str or not (name.startswith('flat_') or name.startswith('bump_')):
                    continue

                pos_x, pos_y, pos_z = map(float, pos_str.split())
                
                x_idx = int(round((pos_x - HALF_SIZE_GRID_XY) / CELL_SIZE))
                y_idx = int(round((pos_y - y_offset_mujoco - HALF_SIZE_GRID_XY) / CELL_SIZE))
                
                if 0 <= x_idx < NUM_CELLS_X and 0 <= y_idx < NUM_CELLS_Y:
                    if name.startswith('flat_'):
                        self.grid[x_idx][y_idx] = 'flat'
                    
                    elif name.startswith('bump_'):
                        if size_str:
                            half_z = float(size_str.split()[2])
                            height_key = bump_half_z_to_key.get(half_z)
                            
                            if height_key:
                                self.grid[x_idx][y_idx] = ('bump', height_key)
                            else:
                                self.grid[x_idx][y_idx] = 'flat' 
                                print(f"Warning: Unrecognized bump ({half_z}), treated as Flat.")

            self.update_canvas_content()
            messagebox.showinfo("Import Successful", 
                                f"File imported successfully!\nGrid updated with corridor data.")

        except FileNotFoundError:
            messagebox.showerror("Error", "File not found.")
        except ET.ParseError:
            messagebox.showerror("Error", "Error parsing XML file. Make sure the format is valid.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred during import:\n{e}")

    def generate_xml(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".xml",
            filetypes=[("MuJoCo XML files", "*.xml")],
            title="Save MuJoCo XML file"
        )
        
        if not filename:
            return

        bump_materials = ""
        for key, settings in BUMP_SETTINGS.items():
            rgb_color = settings["mujoco_rgb"]
            mat_name = settings["mat_name"]
            bump_materials += f'    <material name="{mat_name}" texture="tex_wall" rgba="{rgb_color} 1" specular="0.5" shininess="0.5" />\n'

        xml_template_start = f"""<?xml version='1.0' encoding='utf-8'?>
<mujoco model="corridor_{int(CORRIDOR_WIDTH_M)}x{int(CORRIDOR_LENGTH_M)}">
  <compiler angle="degree" autolimits="true" />
  <option timestep="0.005" gravity="0 0 -9.81" />
  <size njmax="4000" nconmax="1000" />
  <asset>
    <texture name="tex_grid" type="2d" builtin="checker" rgb1="0.1 0.1 0.1" rgb2="0.15 0.15 0.15" width="300" height="300" mark="edge" markrgb="0.8 0.8 0.8" />
    <texture name="tex_wall" type="2d" builtin="checker" rgb1="0.5 0.5 0.5" rgb2="0.55 0.55 0.55" width="300" height="300" />
    <material name="mat_floor" texture="tex_grid" texrepeat="2 2" specular="0.1" shininess="0.1" />
    <material name="mat_wall" texture="tex_wall" texrepeat="1 1" rgba="0.5 0.5 0.5 0" specular="0.3" shininess="0.3" />
{bump_materials.strip()}
  </asset>
  <worldbody>
    <light name="key_light" pos="4 4 4" dir="-1 -1 -1" directional="true" diffuse="1.1 1.1 1.1" specular="0.4 0.4 0.4" castshadow="true" />
    <light name="fill_light" pos="-6 4 2" dir="1 -0.5 -1" directional="true" ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2" castshadow="false" />

    <geom name="wall_left" type="box" size="{CORRIDOR_LENGTH_M/2:.3f} 0.025 4.000" pos="{CORRIDOR_LENGTH_M/2:.3f} {-CORRIDOR_WIDTH_M/2 - 0.025:.3f} 4.000" material="mat_wall" />
    <geom name="wall_right" type="box" size="{CORRIDOR_LENGTH_M/2:.3f} 0.025 4.000" pos="{CORRIDOR_LENGTH_M/2:.3f} {CORRIDOR_WIDTH_M/2 + 0.025:.3f} 4.000" material="mat_wall" />

    """

        xml_geoms = []
        flat_counter = 0
        bump_counter = 0
        
        y_offset_mujoco = -CORRIDOR_WIDTH_M / 2
        HALF_SIZE_GRID_XY = CELL_SIZE / 2.0 
        HALF_SIZE_FLAT_Z = BASE_FLAT_HALF_Z 
        Z_FLOOR_TOP = BASE_FLAT_HALF_Z * 2.0 

        for x_idx in range(NUM_CELLS_X):
            for y_idx in range(NUM_CELLS_Y):
                cell_value = self.grid[x_idx][y_idx]
                cell_type = cell_value[0] if isinstance(cell_value, tuple) else cell_value
                
                center_x = (x_idx * CELL_SIZE) + HALF_SIZE_GRID_XY
                center_y = (y_idx * CELL_SIZE) + HALF_SIZE_GRID_XY + y_offset_mujoco

                if cell_type == 'flat':
                    geom_line = f'    <geom type="box" material="mat_floor" size="{HALF_SIZE_GRID_XY:.3f} {HALF_SIZE_GRID_XY:.3f} {HALF_SIZE_FLAT_Z:.3f}" pos="{center_x:.3f} {center_y:.3f} {HALF_SIZE_FLAT_Z:.3f}" name="flat_{flat_counter}" />'
                    xml_geoms.append(geom_line)
                    flat_counter += 1

                elif cell_type == 'bump':
                    height_key = cell_value[1]
                    settings = BUMP_SETTINGS[height_key]
                    
                    HALF_SIZE_BUMP_Z = settings["half_z"]
                    MAT_NAME = settings["mat_name"]
                    
                    CENTER_Z_BUMP = Z_FLOOR_TOP + HALF_SIZE_BUMP_Z

                    geom_line = f'    <geom type="box" material="{MAT_NAME}" size="{HALF_SIZE_GRID_XY:.3f} {HALF_SIZE_GRID_XY:.3f} {HALF_SIZE_BUMP_Z:.3f}" pos="{center_x:.3f} {center_y:.3f} {CENTER_Z_BUMP:.3f}" name="bump_{bump_counter}" />'
                    xml_geoms.append(geom_line)
                    bump_counter += 1

        xml_template_end = """
  </worldbody>
</mujoco>"""

        full_xml = xml_template_start + "\n" + "\n".join(xml_geoms) + xml_template_end
        
        try:
            with open(filename, 'w') as f:
                f.write(full_xml)
            messagebox.showinfo("Generation Successful", 
                                f"XML file generated successfully!\n\nLocation:\n{filename}\n\nStatistics:\n• {flat_counter} floor cells\n• {bump_counter} bumps")
        except Exception as e:
            messagebox.showerror("Error", f"Error writing file:\n{e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = CorridorEditor(root)
    root.mainloop()