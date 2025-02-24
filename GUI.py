import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog

default_user_config = {
    "is_remote": (False, "Push video to remote?"),
    "video_source": ("0", "Video source?"),
    "use_mmpose_visualizer": (False, "Use MMPOSE visualizer?"),
    "use_trained_yolo": (True, "Use self-trained YOLO model?"),
}


def getUserGuiConfig(default_config):

    def on_submit():
        nonlocal default_config, tk_vars, user_config
        for var_key, (var_value, _) in default_config.items():
            if isinstance(var_value, bool):
                user_config[var_key] = tk_vars[f"{var_key}_var"].get()
            else:
                user_config[var_key] = tk_vars[f"{var_key}_entry"].get()
        root.quit()

    # Init main window
    root = tk.Tk()
    root.title("Posture MMPose Configuration")
    root.config(padx=10, pady=10)

    tk_vars = {}
    user_config = {}

    for key, (default_value, desc) in default_config.items():

        if isinstance(default_value, bool):
            var = tk.BooleanVar(value=default_value)
            tk_vars[f"{key}_var"] = var
            tk.Checkbutton(root, text=desc, variable=var).pack(anchor=tk.W)

        elif isinstance(default_value, str):
            entry = tk.Entry(root, width=30)
            entry.insert(0, default_value)
            tk_vars[f"{key}_entry"] = entry
            tk.Label(root, text=desc).pack(anchor=tk.W)
            entry.pack(anchor=tk.W)

        else:
            raise ValueError("Invalid configuration.")

    submit_button = tk.Button(root, text="Start", command=on_submit)
    submit_button.pack(pady=20)
    root.mainloop()

    return user_config


if __name__ == '__main__':
    config = getUserGuiConfig(default_user_config)
    for key, value in config.items():
        print(f"{key} - {value}")
