import tkinter as tk

default_user_config = {
    "is_remote": (False, "Push video to remote?"),
    "video_source": ("0", "Video source?"),
    "face_announce_interval": (5, "Face announce interval?"),
    "use_mmpose_visualizer": (False, "Use MMPOSE visualizer?"),
    "use_trained_yolo": (True, "Use self-trained YOLO model?"),
    "generate_report": (False, "Generate report?"),
}

dtype_to_tk = {
    bool: "var",
    str: "entry",
    int: "spinbox"
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
        for config_name, (config_value, _) in default_config.items():
            user_config[config_name] = type(config_value)(     # Force convert to target type.
                tk_vars[f"{config_name}_{dtype_to_tk[type(config_value)]}"].get()  # Suffix is determined by type
            )
        root.quit()

    # Init main window
    root = tk.Tk()
    root.title("Posture MMPose Configuration")
    root.config(padx=10, pady=10)

    tk_vars = {}
    user_config = {}

    for name, (default_value, desc) in default_config.items():

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

        else:
            raise ValueError(f"Invalid configuration datatype '{type(default_value)}'.")

    # Submit Button
    submit_button = tk.Button(root, text="Start", command=on_submit)
    submit_button.pack(pady=20)

    root.mainloop()

    return user_config


if __name__ == '__main__':
    config = getUserGuiConfig(default_user_config)
    for key, value in config.items():
        print(f"{key} - {value}, {type(value)}")
