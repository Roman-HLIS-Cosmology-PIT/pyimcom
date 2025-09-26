"""
Context manager for writing report figures.
"""


class ReportFigContext:
    """
    This is a context manager for report figures in PyIMCOM.

    Parameters
    ----------
    mpl : class
        The matplotlib module.
    plt : class
        The pyplot submodule.

    Attributes
    ----------
    mpl : class
        The matplotlib module.
    plt : class
        The pyplot submodule.
    env_backend : str
        The matplotlib backend in the environment.
    usetex : bool or None
        Enviroment self.usetex setting.

    Methods
    -------
    __init__
        Constructor.
    __enter__
        Set up local configuration.
    __exit__
        Restore old settings.

    """

    def __init__(self, mpl, plt):
        self.mpl = mpl
        self.plt = plt

    def __enter__(self):
        self.env_backend = self.mpl.get_backend()
        self.usetex = self.plt.rcParams.get("text.usetex", None)

        self.mpl.use("Agg")
        self.plt.switch_backend("Agg")
        self.plt.rcParams["text.usetex"] = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mpl.use(self.env_backend)
        self.plt.switch_backend(self.env_backend)
        if self.usetex is None:
            if hasattr(self.plt.rcParams, "text.usetex"):  # this will always be true
                del self.plt.rcParams["text.usetex"]
        else:
            self.plt.rcParams["text.usetex"] = self.usetex
