""" Contains container for storing dataset of seismic crops. """
import dill

import numpy as np
import matplotlib.pyplot as plt

from ..batchflow import Dataset, Sampler
from ..batchflow import HistoSampler, NumpySampler, ConstantSampler
from .geometry import SeismicGeometry
from .crop_batch import SeismicCropBatch

from .utils import read_point_cloud, make_labels_dict, _get_horizons, round_to_array


class SeismicCubeset(Dataset):
    """ Stores indexing structure for dataset of seismic cubes along with additional structures.
    """
    def __init__(self, index, batch_class=SeismicCropBatch, preloaded=None, *args, **kwargs):
        """ Initialize additional attributes.
        """
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, *args, **kwargs)
        self.geometries = {ix: SeismicGeometry(self.index.get_fullpath(ix)) for ix in self.indices}
        self.point_clouds = {ix: np.array([]) for ix in self.indices}
        self.labels = {ix: dict() for ix in self.indices}
        self.samplers = {ix: None for ix in self.indices}
        self.sampler = None

        self.grid_gen, self.grid_info, self.grid_iters = None, None, None


    def load_geometries(self, path=None, logs=True):
        """ Load geometries into dataset-attribute.

        Parameters
        ----------
        load_from : str
            Path to load geometries from.

        scalers : bool
            Whether to make callables to scale initial values in .sgy cube to
            [0, 1] range. It takes quite a lot of time.

        mode : one of 'full', 'random'
            Sampler creating mode. Determines amount of trace to check to find
            cube minimum/maximum values.

        logs : bool
            Whether to create logs. If True, .log file is created next to .sgy-cube location.

        Returns
        -------
        SeismicCubeset
            Same instance with loaded geometries.
        """
        if isinstance(load_from, str):
            with open(load_from, 'rb') as file:
                self.geometries = dill.load(file)
        else:
            for ix in self.indices:
                self.geometries[ix].load()
                if logs:
                    self.geometries[ix].log()
        return self


    def save_geometries(self, save_to):
        """ Save dill-serialized geometries for a dataset of seismic-cubes on disk.
        """
        if isinstance(save_to, str):
            with open(save_to, 'wb') as file:
                dill.dump(self.geometries, file)
        return self


    def convert_to_h5py(self):
        """ Converts every cube in dataset from `.sgy` to `.hdf5`. """
        for ix in self.indices:
            self.geometries[ix].make_h5py()
        return self


    def load_point_clouds(self, paths=None, load_from=None, **kwargs):
        """ Load point-cloud of labels for each cube in dataset.

        Parameters
        ----------
        paths : dict
            Mapping from indices to txt paths with labels.

        load_from : str
            Path to load point clouds from.

        Returns
        -------
        SeismicCubeset
            Same instance with loaded point clouds.
        """
        if isinstance(load_from, str):
            with open(load_from, 'rb') as file:
                self.point_clouds = dill.load(file)
        else:
            for ix in self.indices:
                self.point_clouds[ix] = read_point_cloud(paths[ix], **kwargs)
        return self


    def save_point_clouds(self, save_to):
        """ Save dill-serialized point clouds for a dataset of seismic-cubes on disk.
        """
        if isinstance(save_to, str):
            with open(save_to, 'wb') as file:
                dill.dump(self.point_clouds, file)
        return self


    def load_labels(self, path=None, transforms=None, src='point_clouds', dst='labels'):
        """ Make labels in inline-xline coordinates using cloud of points and supplied transforms.

        Parameters
        ----------
        load_from : str
            Path to load labels from.

        transforms : dict
            Mapping from indices to callables. Each callable should define
            way to map point from absolute coordinates (X, Y world-wise) to
            cube specific (ILINE, XLINE) and take array of shape (N, 4) as input.

        src : str
            Attribute with saved point clouds.

        Returns
        SeismicCubeset
            Same instance with loaded labels.
        """
        point_clouds = getattr(self, src) if isinstance(src, str) else src
        transforms = transforms or dict()
        if not hasattr(self, dst):
            setattr(self, dst, {})

        if isinstance(load_from, str):
            try:
                with open(path, 'rb') as file:
                    setattr(self, dst, dill.load(file))
            except TypeError:
                raise NotImplementedError("Numba dicts are yet to support serializing")
        else:
            for ix in self.indices:
                point_cloud = point_clouds.get(ix)
                geom = getattr(self, 'geometries').get(ix)
                transform = transforms.get(ix) or geom.abs_to_lines
                getattr(self, dst)[ix] = make_labels_dict(transform(point_cloud))
        return self


    def save_labels(self, save_to, src='labels'):
        """ Save dill-serialized labels for a dataset of seismic-cubes on disk. """
        if isinstance(save_to, str):
            try:
                with open(save_to, 'wb') as file:
                    dill.dump(getattr(self, src), file)
            except TypeError:
                raise NotImplementedError("Numba dicts are yet to support serializing")
        return self


    def load_samplers(self, path=None, mode='hist', p=None,
                      transforms=None, dst='sampler', **kwargs):
        """ Create samplers for every cube and store it in `samplers`
        attribute of passed dataset. Also creates one combined sampler
        and stores it in `sampler` attribute of passed dataset.

        Parameters
        ----------
        load_from : str
            Path to load samplers from.

        mode : str or Sampler
            Type of sampler to be created.
            If 'hist', then sampler is estimated from given labels.
            If 'numpy', then sampler is created with `kwargs` parameters.
            If instance of Sampler is provided, it must generate points from unit cube.

        p : list
            Weights for each mixture in final sampler.

        transforms : dict
            Mapping from indices to callables. Each callable should define
            way to map point from absolute coordinates (X, Y world-wise) to
            cube local specific and take array of shape (N, 4) as input.

        Note
        ----
        Passed `dataset` must have `geometries` and `labels` attributes if
        you want to create HistoSampler.
        """
        #pylint: disable=cell-var-from-loop
        lowcut, highcut = [0, 0, 0], [1, 1, 1]
        transforms = transforms or dict()

        if isinstance(load_from, str):
            with open(load_from, 'rb') as file:
                samplers = dill.load(file)

        else:
            samplers = {}
            if not isinstance(mode, dict):
                mode = {ix: mode for ix in self.indices}

            for ix in self.indices:
                if isinstance(mode[ix], Sampler):
                    sampler = mode[ix]
                elif mode[ix] == 'numpy':
                    sampler = NumpySampler(**kwargs)
                elif mode[ix] == 'hist':
                    point_cloud = getattr(self, 'point_clouds')[ix]

                    geom = getattr(self, 'geometries')[ix]
                    offsets = np.array([geom.ilines_offset, geom.xlines_offset, 0])
                    cube_shape = np.array(geom.cube_shape)
                    to_cube = lambda points: (points[:, :3] - offsets)/cube_shape
                    default = lambda points: to_cube(geom.abs_to_lines(points))

                    transform = transforms.get(ix) or default
                    cube_array = transform(point_cloud)

                    bins = kwargs.get('bins') or 100
                    sampler = HistoSampler(np.histogramdd(cube_array, bins=bins))
                else:
                    sampler = NumpySampler('u', low=0, high=1, dim=3)

                sampler = sampler.truncate(low=lowcut, high=highcut)
                samplers.update({ix: sampler})
        self.samplers = samplers

        # One sampler to rule them all
        p = p or [1/len(self) for _ in self.indices]

        sampler = 0 & NumpySampler('n', dim=4)
        for i, ix in enumerate(self.indices):
            sampler_ = (ConstantSampler(ix)
                        & samplers[ix].apply(lambda d: d.astype(np.object)))
            sampler = sampler | (p[i] & sampler_)
        setattr(self, dst, sampler)
        return self


    def modify_sampler(self, dst, mode='iline', low=None, high=None,
                       each=None, each_start=None,
                       to_cube=False, post=None, finish=False, src='sampler'):
        """ Change given sampler to generate points from desired regions.

        Parameters
        ----------
        src : str
            Attribute with Sampler to change.

        dst : str
            Attribute to store created Sampler.

        mode : str
            Axis to modify: ilines/xlines/heights.

        low : float
            Lower bound for truncating.

        high : float
            Upper bound for truncating.

        each : int
            Keep only i-th value along axis.

        each_start : int
            Shift grid for previous parameter.

        to_cube : bool
            Transform sampled values to each cube coordinates.

        post : callable
            Additional function to apply to sampled points.

        finish : bool
            If False, instance of Sampler is put into `dst` and can be modified later.
            If True, `sample` method is put into `dst` and can be called via `D` named-expressions.

        Examples
        --------
        Split into train / test along ilines in 80/20 ratio:

        >>> cubeset.modify_sampler(dst='train_sampler', mode='i', high=0.8)
        >>> cubeset.modify_sampler(dst='test_sampler', mode='i', low=0.9)

        Sample only every 50-th point along xlines starting from 70-th xline:

        >>> cubeset.modify_sampler(dst='train_sampler', mode='x', each=50, each_start=70)

        Notes
        -----
        It is advised to have gap between `high` for train sampler and `low` for test sampler.
        That is done in order to take into account additional seen entries due to crop shape.
        """

        # Parsing arguments
        sampler = getattr(self, src)

        mapping = {'ilines': 0, 'xlines': 1, 'heights': 2,
                   'iline': 0, 'xline': 1, 'i': 0, 'x': 1, 'h': 2}
        axis = mapping[mode]

        low, high = low or 0, high or 1
        each_start = each_start or each

        # Keep only points from region
        if (low != 0) or (high != 1):
            sampler = sampler.truncate(low=low, high=high, prob=high-low,
                                       expr=lambda p: p[:, axis+1])

        # Keep only every `each`-th point
        if each is not None:
            def get_shape(name):
                return self.geometries[name].cube_shape[axis]

            def get_ticks(name):
                shape = self.geometries[name].cube_shape[axis]
                return np.arange(each_start, shape, each)

            def filter_out(array):
                shapes = np.array(list(map(get_shape, array[:, 0])))
                ticks = np.array(list(map(get_ticks, array[:, 0])))
                arr = (array[:, axis+1]*shapes).astype(int)
                array[:, axis+1] = round_to_array(arr, ticks) / shapes
                return array

            sampler = sampler.apply(filter_out)

        # Change representation of points from unit cube to cube coordinates
        if to_cube:
            def get_shapes(name):
                return self.geometries[name].cube_shape

            def coords_to_cube(array):
                shapes = np.array(list(map(get_shapes, array[:, 0])))
                array[:, 1:] = (array[:, 1:] * shapes).astype(int)
                return array

            sampler = sampler.apply(coords_to_cube)

        # Apply additional transformations to points
        if callable(post):
            sampler = sampler.apply(post)

        if finish:
            setattr(self, dst, sampler.sample)
        else:
            setattr(self, dst, sampler)


    def save_samplers(self, save_to):
        """ Save dill-serialized samplers for a dataset of seismic-cubes on disk.
        """
        if isinstance(save_to, str):
            with open(save_to, 'wb') as file:
                dill.dump(self.samplers, file)
        return self


    def save_attr(self, name, save_to):
        """ Save attribute of dataset to disk. """
        if isinstance(save_to, str):
            with open(save_to, 'wb') as file:
                dill.dump(getattr(self, name), file)
        return self


    def make_grid(self, cube_name, crop_shape,
                  ilines_range, xlines_range, h_range,
                  strides=None, batch_size=16):
        """ Create regular grid of points in cube.
        This method is usually used with `assemble_predict` action of SeismicCropBatch.

        Parameters
        ----------
        cube_name : str
            Reference to cube. Should be valid key for `geometries` attribute.

        crop_shape : array-like
            Shape of model inputs.

        ilines_range : array-like of two elements
            Location of desired prediction, iline-wise.

        xlines_range : array-like of two elements
            Location of desired prediction, xline-wise.

        h_range : array-like of two elements
            Location of desired prediction, depth-wise.

        strides : array-like
            Distance between grid points.

        batch_size : int
            Amount of returned points per generator call.

        Returns
        -------
        SeismicCubeset
            Same instance with grid generator and grid information in attributes.

        """
        geom = self.geometries[cube_name]
        strides = strides or crop_shape

        # Assert ranges are valid
        if ilines_range[0] < 0 or \
           xlines_range[0] < 0 or \
           h_range[0] < 0:
            raise ValueError('Ranges must contain in the cube.')

        if ilines_range[1] >= geom.ilines_len or \
           xlines_range[1] >= geom.xlines_len or \
           h_range[1] >= geom.depth:
            raise ValueError('Ranges must contain in the cube.')

        # Make separate grids for every axis
        def _make_axis_grid(axis_range, stride, length, crop_shape):
            grid = np.arange(*axis_range, stride)
            grid_ = [x for x in grid if x + crop_shape < length]
            if len(grid) != len(grid_):
                grid_ += [axis_range[1] - crop_shape]
            return sorted(grid_)

        ilines = _make_axis_grid(ilines_range, strides[0], geom.ilines_len, crop_shape[0])
        xlines = _make_axis_grid(xlines_range, strides[1], geom.xlines_len, crop_shape[1])
        hs = _make_axis_grid(h_range, strides[2], geom.depth, crop_shape[2])

        # Every point in grid contains reference to cube
        # in order to be valid input for `crop` action of SeismicCropBatch
        grid = []
        for il in ilines:
            for xl in xlines:
                for h in hs:
                    point = [cube_name, il, xl, h]
                    grid.append(point)
        grid = np.array(grid, dtype=object)

        # Creating and storing all the necessary things
        grid_gen = (grid[i:i+batch_size]
                    for i in range(0, len(grid), batch_size))

        offsets = np.array([min(grid[:, 1]),
                            min(grid[:, 2]),
                            min(grid[:, 3])])

        predict_shape = (ilines_range[1] - ilines_range[0],
                         xlines_range[1] - xlines_range[0],
                         h_range[1] - h_range[0])

        grid_array = grid[:, 1:].astype(int) - offsets

        self.grid_gen = lambda: next(grid_gen)
        self.grid_iters = - (-len(grid) // batch_size)
        self.grid_info = {'grid_array': grid_array,
                          'predict_shape': predict_shape,
                          'crop_shape': crop_shape,
                          'cube_name': cube_name,
                          'range': [ilines_range, xlines_range, h_range]}
        return self

    def get_point_cloud(self, src, dst, threshold=0.5, averaging='mean', coordinates='cubic', separate=True):
        """ Compute point cloud of horizons from a mask, save it into the 'cubeset'-attribute.

        Parameters
        ----------
        src : str or array
            Source-mask. Can be either a name of attribute or mask itself.

        dst : str
            Attribute of `cubeset` to write the horizons in.

        threshold : float
            Parameter of mask-thresholding.

        averaging : str
            Method of pandas.groupby used for finding the center of a horizon
            for each (iline, xline).

        coordinates : str
            Coordinates to use for keys of point-cloud. Can be either 'cubic'
            'lines' or None. In case of None, mask-coordinates are used. Mode 'cubic'
            requires 'grid_info'-attribute; can be run after `make_grid`-method. Mode 'lines'
            requires both 'grid_info' and 'geometries'-attributes to be loaded.

        separate : bool
            Whether to write horizonts in separate dictionaries or in one common.

        Returns
        -------
        dict or list of dict
            If separate is False, then one dictionary returned with keys being pairs of
            (iline, xline) and values being lists of heights.
            If separate is True, then list of dictionaries is returned, with every dictionary being
            mapping from pairs of (iline, xline) to height from each individual horizont.
        """
        # fetch mask-array
        mask = getattr(self, src) if isinstance(src, str) else src

        # prepare coordinate-transforms
        if coordinates is None:
            transforms = [lambda x: x for _ in range(3)]
        elif coordinates == 'cubic':
            shifts = [axis_range[0] for axis_range in self.grid_info['range']]
            transforms = [lambda x_, shift=shift: x_ + shift for shift in shifts]
        elif coordinates == 'lines':
            geom = self.geometries[self.grid_info['cube_name']]
            i_shift, x_shift, h_shift = [axis_range[0] for axis_range in self.grid_info['range']]
            transforms = [lambda i_: geom.ilines[i_ + i_shift], lambda x_: geom.xlines[x_ + x_shift],
                          lambda h_: h_ + h_shift]

        # get horizons
        setattr(self, dst, _get_horizons(mask, threshold, averaging, transforms, separate))

        if separate:
            horizons = getattr(self, dst)
            horizons.sort(key=len, reverse=True)
            for i, horizon in enumerate(horizons):
                setattr(self, dst+'_'+str(i), horizon)
        return self
