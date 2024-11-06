import _plotly_utils
from _plotly_utils.basevalidators import get_module, warnings, is_numpy_convertable, to_scalar_or_list
import plotly
from plotly.basedatatypes import BaseFigure, deepcopy, OrderedDict, _check_path_in_prop_tree
import plotly.basedatatypes


def copy_to_readonly_numpy_array(v, kind=None, force_numeric=False):
    """
    Convert an array-like value into a read-only numpy array

    Parameters
    ----------
    v : array like
        Array like value (list, tuple, numpy array, pandas series, etc.)
    kind : str or tuple of str
        If specified, the numpy dtype kind (or kinds) that the array should
        have, or be converted to if possible.
        If not specified then let numpy infer the datatype
    force_numeric : bool
        If true, raise an exception if the resulting numpy array does not
        have a numeric dtype (i.e. dtype.kind not in ['u', 'i', 'f'])
    Returns
    -------
    np.ndarray
        Numpy array with the 'WRITEABLE' flag set to False
    """
    np = get_module("numpy")

    # Don't force pandas to be loaded, we only want to know if it's already loaded
    pd = get_module("pandas", should_load=False)
    assert np is not None

    # ### Process kind ###
    if not kind:
        kind = ()
    elif isinstance(kind, str):
        kind = (kind,)

    first_kind = kind[0] if kind else None

    # u: unsigned int, i: signed int, f: float
    numeric_kinds = {"u", "i", "f"}
    kind_default_dtypes = {
        "u": "uint32",
        "i": "int32",
        "f": "float64",
        "O": "object",
    }

    # Handle pandas Series and Index objects
    if pd and isinstance(v, (pd.Series, pd.Index)):
        if v.dtype.kind in numeric_kinds:
            # Get the numeric numpy array so we use fast path below
            v = v.values
        elif v.dtype.kind == "M":
            # Convert datetime Series/Index to numpy array of datetimes
            if isinstance(v, pd.Series):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    # Series.dt.to_pydatetime will return Index[object]
                    # https://github.com/pandas-dev/pandas/pull/52459
                    v = np.array(v.dt.to_pydatetime())
            else:
                # DatetimeIndex
                v = v.to_pydatetime()
    elif pd and isinstance(v, pd.DataFrame) and len(set(v.dtypes)) == 1:
        dtype = v.dtypes.tolist()[0]
        if dtype.kind in numeric_kinds:
            v = v.values
        elif dtype.kind == "M":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                # Series.dt.to_pydatetime will return Index[object]
                # https://github.com/pandas-dev/pandas/pull/52459
                v = [np.array(row.dt.to_pydatetime()).tolist() for i, row in v.iterrows()]

    if not isinstance(v, np.ndarray):
        # v has its own logic on how to convert itself into a numpy array
        if is_numpy_convertable(v):
            return copy_to_readonly_numpy_array(np.array(v), kind=kind, force_numeric=force_numeric)
        else:
            # v is not homogenous array
            v_list = [to_scalar_or_list(e) for e in v]

            # Lookup dtype for requested kind, if any
            dtype = kind_default_dtypes.get(first_kind, None)

            # construct new array from list
            new_v = np.array(v_list, order="C", dtype=dtype)
    elif v.dtype.kind in numeric_kinds:
        # v is a homogenous numeric array
        if kind and v.dtype.kind not in kind:
            # Kind(s) were specified and this array doesn't match
            # Convert to the default dtype for the first kind
            dtype = kind_default_dtypes.get(first_kind, None)
            new_v = np.ascontiguousarray(v.astype(dtype))
        else:
            # Either no kind was requested or requested kind is satisfied
            if v.flags["C_CONTIGUOUS"]:
                new_v = v
            else:
                new_v = np.ascontiguousarray(v.copy())
    else:
        # v is a non-numeric homogenous array
        new_v = v.copy()

    # Handle force numeric param
    # --------------------------
    if force_numeric and new_v.dtype.kind not in numeric_kinds:
        raise ValueError("Input value is not numeric and force_numeric parameter set to True")

    if "U" not in kind:
        # Force non-numeric arrays to have object type
        # --------------------------------------------
        # Here we make sure that non-numeric arrays have the object
        # datatype. This works around cases like np.array([1, 2, '3']) where
        # numpy converts the integers to strings and returns array of dtype
        # '<U21'
        if new_v.dtype.kind not in ["u", "i", "f", "O", "M"]:
            new_v = np.array(v, dtype="object")

    # Set new array to be read-only
    # -----------------------------
    new_v.flags["WRITEABLE"] = False

    return new_v


_plotly_utils.basevalidators.copy_to_readonly_numpy_array = copy_to_readonly_numpy_array


def base_figure__init__(self, data=None, layout_plotly=None, frames=None, skip_invalid=False, **kwargs):
    from plotly.validators import DataValidator, LayoutValidator, FramesValidator

    super(BaseFigure, self).__init__()

    # Initialize validation
    self._validate = kwargs.pop("_validate", True)

    # Assign layout_plotly to layout
    # ------------------------------
    # See docstring note for explanation
    layout = layout_plotly

    # Subplot properties
    # ------------------
    # These properties are used by the tools.make_subplots logic.
    # We initialize them to None here, before checking if the input data
    # object is a BaseFigure, or a dict with _grid_str and _grid_ref
    # properties, in which case we bring over the _grid* properties of
    # the input
    self._grid_str = None
    self._grid_ref = None

    # Handle case where data is a Figure or Figure-like dict
    # ------------------------------------------------------
    if isinstance(data, BaseFigure):
        # Bring over subplot fields
        self._grid_str = data._grid_str
        self._grid_ref = data._grid_ref

        # Extract data, layout, and frames
        data, layout, frames = data.data, data.layout, data.frames

    elif isinstance(data, dict) and ("data" in data or "layout" in data or "frames" in data):
        # Bring over subplot fields
        self._grid_str = data.get("_grid_str", None)
        self._grid_ref = data.get("_grid_ref", None)

        # Extract data, layout, and frames
        data, layout, frames = (
            data.get("data", None),
            data.get("layout", None),
            data.get("frames", None),
        )

    # Handle data (traces)
    # --------------------
    # ### Construct data validator ###
    # This is the validator that handles importing sequences of trace
    # objects
    self._data_validator = DataValidator(set_uid=self._set_trace_uid)

    # ### Import traces ###
    data = self._data_validator.validate_coerce(data, skip_invalid=skip_invalid, _validate=self._validate)

    # ### Save tuple of trace objects ###
    self._data_objs = data

    # ### Import clone of trace properties ###
    # The _data property is a list of dicts containing the properties
    # explicitly set by the user for each trace.
    # self._data = [deepcopy(trace._props) for trace in data]
    self._data = [trace._props for trace in data]

    # ### Create data defaults ###
    # _data_defaults is a tuple of dicts, one for each trace. When
    # running in a widget context, these defaults are populated with
    # all property values chosen by the Plotly.js library that
    # aren't explicitly specified by the user.
    #
    # Note: No property should exist in both the _data and
    # _data_defaults for the same trace.
    self._data_defaults = [{} for _ in data]

    # ### Reparent trace objects ###
    for trace_ind, trace in enumerate(data):
        # By setting the trace's parent to be this figure, we tell the
        # trace object to use the figure's _data and _data_defaults
        # dicts to get/set it's properties, rather than using the trace
        # object's internal _orphan_props dict.
        trace._parent = self

        # We clear the orphan props since the trace no longer needs then
        # trace._orphan_props.clear()

        # Set trace index
        trace._trace_ind = trace_ind

    # Layout
    # ------
    # ### Construct layout validator ###
    # This is the validator that handles importing Layout objects
    self._layout_validator = LayoutValidator()

    # ### Import Layout ###
    self._layout_obj = self._layout_validator.validate_coerce(
        layout, skip_invalid=skip_invalid, _validate=self._validate
    )

    # ### Import clone of layout properties ###
    self._layout = deepcopy(self._layout_obj._props)

    # ### Initialize layout defaults dict ###
    self._layout_defaults = {}

    # ### Reparent layout object ###
    self._layout_obj._orphan_props.clear()
    self._layout_obj._parent = self

    # Config
    # ------
    # Pass along default config to the front end. For now this just
    # ensures that the plotly domain url gets passed to the front end.
    # In the future we can extend this to allow the user to supply
    # arbitrary config options like in plotly.offline.plot/iplot.  But
    # this will require a fair amount of testing to determine which
    # options are compatible with FigureWidget.
    from plotly.offline.offline import _get_jconfig

    self._config = _get_jconfig(None)

    # Frames
    # ------

    # ### Construct frames validator ###
    # This is the validator that handles importing sequences of frame
    # objects
    self._frames_validator = FramesValidator()

    # ### Import frames ###
    self._frame_objs = self._frames_validator.validate_coerce(frames, skip_invalid=skip_invalid)

    # Note: Because frames are not currently supported in the widget
    # context, we don't need to follow the pattern above and create
    # _frames and _frame_defaults properties and then reparent the
    # frames. The figure doesn't need to be notified of
    # changes to the properties in the frames object hierarchy.

    # Context manager
    # ---------------

    # ### batch mode indicator ###
    # Flag that indicates whether we're currently inside a batch_*()
    # context
    self._in_batch_mode = False

    # ### Batch trace edits ###
    # Dict from trace indexes to trace edit dicts. These trace edit dicts
    # are suitable as `data` elements of Plotly.animate, but not
    # the Plotly.update (See `_build_update_params_from_batch`)
    self._batch_trace_edits = OrderedDict()

    # ### Batch layout edits ###
    # Dict from layout properties to new layout values. This dict is
    # directly suitable for use in Plotly.animate and Plotly.update
    self._batch_layout_edits = OrderedDict()

    # Animation property validators
    # -----------------------------
    from plotly import animation

    self._animation_duration_validator = animation.DurationValidator()
    self._animation_easing_validator = animation.EasingValidator()

    # Template
    # --------
    # ### Check for default template ###
    self._initialize_layout_template()

    # Process kwargs
    # --------------
    for k, v in kwargs.items():
        err = _check_path_in_prop_tree(self, k)
        if err is None:
            self[k] = v
        elif not skip_invalid:
            type_err = TypeError("invalid Figure property: {}".format(k))
            type_err.args = (
                type_err.args[0]
                + """
%s"""
                % (err.args[0],),
            )
            raise type_err


plotly.basedatatypes.BaseFigure.__init__ = base_figure__init__
