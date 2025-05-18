import tkinter as tk
from tkinter import ttk
from utils.decorations import CONSOLE_COLORS as CC

default_user_config = {
    "video_source": ("0", "Video source (Camera Index/File Path)"),
    "is_remote": (True, "Push video to remote?"),
    "websocket_url": ("ws://localhost:8976", "Remote source URL"),
    "face_announce_interval": (5, "Face announce interval?"),
    "pose_conf": (0.8, "Posture Confidence"),
    "phone_conf": (0.35, "Phone Confidence"),
    "spare": (0.45, "Spareness"),
    "use_mmpose_visualizer": (False, "Use MMPOSE visualizer?"),
    "use_trained_yolo": (True, "Use self-trained YOLO model?"),
    "generate_report": (False, "Generate report?"),
}

gui_separator = {
    "video_source": "User Options",
    "use_mmpose_visualizer": "Developer Options"
}

dtype_to_tk = {
    bool: "var",
    str: "entry",
    int: "spinbox",
    float: "entry"
}


def getUserGuiConfig(default_config):
    """
    Get user configuration from a GUI panel.
    :param default_config:  Dictionary of default configurations.
    :return: Possibly modified user configuration.
    """

    def on_submit():
        """
        Invoked call-back when the submit button is hit.
        :return: None.
        """
        nonlocal default_config, tk_vars, user_config
        for config_name, (config_value, _, *extra) in default_config.items():
            try:
                user_config[config_name] = type(config_value)(     # Force convert to target type.
                    tk_vars[f"{config_name}_{dtype_to_tk[type(config_value)]}"].get()  # Suffix is determined by type
                )
            except ValueError as e:
                # Users may enter some bad strings into the float string field..... can't convert to float.
                # Just use the default setting instead.
                print(f"{CC['yellow']}"
                      f"[Input Panel] Bad string for field {config_name}, "
                      f"fall back to default value {config_value}."
                      f"{CC['reset']}")
        root.quit()

    # Init main window
    root = tk.Tk()
    root.title("Posture MMPose Configuration")
    root.config(padx=10, pady=10)

    tk_vars = {}
    user_config = {}

    for name, (default_value, desc) in default_config.items():
        # Separator
        if name in gui_separator:
            if desc is not None:
                separator_title = tk.Label(root, text=gui_separator[name], font=("TkDefaultFont", 8, "bold"))
                separator_title.pack(anchor=tk.W, pady=(10, 0))
            separator = ttk.Separator(root, orient="horizontal")
            separator.pack(fill="x", pady=0)

        if isinstance(default_value, bool):
            var = tk.BooleanVar(value=default_value)
            tk_vars[f"{name}_{dtype_to_tk[bool]}"] = var
            tk.Checkbutton(root, text=desc, variable=var).pack(anchor=tk.W)

        elif isinstance(default_value, str):
            entry = tk.Entry(root, width=30)
            entry.insert(0, default_value)
            tk_vars[f"{name}_{dtype_to_tk[str]}"] = entry
            tk.Label(root, text=desc).pack(anchor=tk.W)
            entry.pack(anchor=tk.W)

        elif isinstance(default_value, int):
            spinbox = tk.Spinbox(root, from_=default_value, to=20, width=5, increment=1)
            tk_vars[f"{name}_{dtype_to_tk[int]}"] = spinbox
            tk.Label(root, text=desc).pack(anchor=tk.W)
            spinbox.pack(anchor=tk.W)

        elif isinstance(default_value, float):
            entry = tk.Entry(root, width=30)
            entry.insert(0, str(default_value))
            tk_vars[f"{name}_{dtype_to_tk[float]}"] = entry
            tk.Label(root, text=desc).pack(anchor=tk.W)
            entry.pack(anchor=tk.W)
            pass

        else:
            raise ValueError(f"Invalid configuration datatype '{type(default_value)}'.")

    # Separator before button
    separator = ttk.Separator(root, orient="horizontal")
    separator.pack(fill="x", pady=(10, 0))

    # Submit Button
    submit_button = tk.Button(root, text="Start", command=on_submit)
    submit_button.pack(pady=(10, 0))

    root.mainloop()

    return user_config


if __name__ == '__main__':
    config = getUserGuiConfig(default_user_config)
    for key, value in config.items():
        print(f"{key} - {value}, {type(value)}")
