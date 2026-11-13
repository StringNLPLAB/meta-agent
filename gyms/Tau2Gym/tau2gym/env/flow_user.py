"""Legacy flow is intentionally absent; use the editable CoSER bootstrap bundle."""
class Tau2FlowUser:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Configure EAU_USER_SIM_BUNDLE or native user mode")
