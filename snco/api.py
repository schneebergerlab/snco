import numpy as np

from snco.records import MarkerRecords, PredictionRecords
from snco.plot import single_cell_markerplot, plot_recombination_landscape, plot_allele_ratio


class RecordsAPIMixin:

    '''
    Mixin adding useful API functions to Records objects, mostly for use in IPython
    '''

    def __getattr__(self, attribute):
        '''makes metadata keys available as attributes'''
        try:
            return self.metadata[attribute]
        except KeyError:
            raise AttributeError(f"'{type(self)}' object has no attribute '{attribute}'")

    def __dir__(self):
        '''makes metadata keys available in __dir__ so that IPython attribute tab completion includes them'''
        return object.__dir__(self) + list(self.metadata.keys())

    def _ipython_key_completions_(self):
        '''makes top level keys available in IPython'''
        return self._records.keys()

    def _repr_table_info(self):
        raise NotImplementedError() 

    def _repr_html_(self):
        '''produces a prettier html table representation of Records objects'''
        n_cb = len(self)
        n_chroms = len(self.chrom_sizes)
        cls_name = self.__class__.__qualname__

        rows, stat_name = self._repr_table_info()
        records_info = f"""
        <div style="font-family: sans-serif">
            <p>{cls_name} object with <strong>{n_cb}</strong> barcodes across <strong>{n_chroms}</strong> chromosomes.</p>
            <table border="1" cellpadding="4" cellspacing="0">
                <thead>
                    <tr>
                        <th>Cell barcode</th>
                        <th>{stat_name}</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """
        return records_info


class MarkerRecordsWrapper(MarkerRecords, RecordsAPIMixin):

    __qualname__ = 'MarkerRecords'

    def plot_barcode(self, cb, **kwargs):
        """
        Plot a single-cell marker profile.
    
        Parameters
        ----------
        cb : str
            Cell barcode to plot.
        co_preds : PredictionRecords, optional
            PredictionRecords object containing the haplotype predictions. Default is None.
        figsize : tuple, optional
            The size of the figure (width, height) in inches. Default is (18, 4).
        chroms : list of str, optional
            A list of chromosome names to plot. If None, all chromosomes will be plotted. Default is None.
        show_mesh_prob : bool, optional
            Whether to display the haplotype probability mesh. Default is True.
        annotate_co_number : bool, optional
            Whether to annotate the number of crossovers for each chromosome. Default is True.
        nco_min_prob_change : float, optional
            The minimum probability change to consider when counting crossovers for annotation. Default is 5e-3.
        show_gt : bool, optional
            Whether to show the ground truth crossover locations for simulated data, where available. Default is True.
        max_yheight : float or 'auto', optional
            The maximum y-axis height for the plots. If 'auto', the 99.5th percentile of all marker values 
            is used. Default is 'auto'.
        ref_colour : str, optional
            The color for the reference markers. Default is '#0072b2'.
        alt_colour : str, optional
            The color for the alternate markers. Default is '#d55e00'.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object generated by the plot.
        axes : np.ndarray of matplotlib.axes.Axes
            1D numpy array of axes objects for the subplots.

        See Also
        --------
        snco.plot.single_cell_markerplot : The underlying plotting function with
            full customization options.
        """
        return single_cell_markerplot(cb, self, **kwargs)

    def _repr_table_info(self):
        rows = []
        for cb in self.barcodes[:10]:
            # table shows counts per chromosome
            cb_info = ', '.join(f'{chrom}: {int(self[cb, chrom].sum())}' for chrom in self.chrom_sizes)
            rows.append(f"<tr><td>{cb}</td><td>{cb_info}</td></tr>")
        return ''.join(rows), 'Marker counts'


class PredictionRecordsWrapper(PredictionRecords, RecordsAPIMixin):

    __qualname__ = 'PredictionRecords'

    def plot_recombination_landscape(self, **kwargs):
        """
        Plot the recombination landscape across chromosomes for multiple cell barcodes.
    
        Parameters
        ----------
        co_markers : MarkerRecords, optional
            MarkerRecords object containing the marker data for the dataset. When provided, used for calculating edge
            effects only at chromosome ends only. Default is None.
        cb_whitelist : list of str, optional
            A list of cell barcodes to include in the analysis. If None, all barcodes are included. Default is None.
        rolling_mean_window_size : int, optional
            The size of the window for computing the rolling mean (in base pairs). Default is 1,000,000.
        nboots : int, optional
            The number of bootstrap iterations for calculating confidence intervals. Default is 100.
        ci : int, optional
            The confidence interval percentage. Default is 95.
        min_prob : float, optional
            The minimum probability change to consider when counting crossovers. Default is 5e-3.
        axes : list of matplotlib.axes.Axes, optional
            The axes to plot on. If None, new axes are created. Default is None.
        figsize : tuple, optional
            The size of the figure (width, height) in inches. Default is (12, 4).
        colour : str, optional
            The colour to use for the plot lines and fills. If None, a colour is selected from the default palette.
            Default is None.
        label : str, optional
            The label for the plot legend. If None, no legend is added. Default is None.
        rng : numpy.random.Generator, optional
            The random number generator to use for bootstrapping. Default is `DEFAULT_RNG`.
    
        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure containing the recombination landscape plots.
        axes : list of matplotlib.axes.Axes
            The axes containing the recombination landscape plots.

        See Also
        --------
        snco.plot.plot_recombination_lanscape : The underlying plotting function with
            full customization options.
        """
        return plot_recombination_landscape(self, **kwargs)

    def plot_allele_ratio(self, **kwargs):
        """
        Plot the allele ratio across chromosomes for multiple cells with bootstrapped confidence intervals.

        Parameters
        ----------
        cb_whitelist : list of str, optional
            A list of cell barcodes to include in the analysis. If None, all barcodes are included. Default is None.
        nboots : int, optional
            The number of bootstrap iterations for calculating confidence intervals. Default is 100.
        ci : int, optional
            The confidence interval percentage. Default is 95.
        axes : list of matplotlib.axes.Axes, optional
            The axes to plot on. If None, new axes are created. Default is None.
        figsize : tuple, optional
            The size of the figure (width, height) in inches. Default is (12, 4).
        colour : str, optional
            The colour to use for the plot lines and fills. If None, a colour is selected from the default palette.
            Default is None.
        label : str, optional
            The label for the plot legend. If None, no legend is added. Default is None.
        rng : numpy.random.Generator, optional
            The random number generator to use for bootstrapping. Default is `DEFAULT_RNG`.
    
        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure containing the marker coverage plots.
        axes : list of matplotlib.axes.Axes
            The axes containing the allele ratio plots.

        See Also
        --------
        snco.plot.plot_allele_ratio : The underlying plotting function with
            full customization options.
        """
        return plot_allele_ratio(self, **kwargs)
    
    def _repr_table_info(self):
        rows = []
        for cb in self.barcodes[:10]:
            # table shows estimated crossovers per chromosome
            cb_info = []
            for chrom in self.chrom_sizes:
                hp = self[cb, chrom]
                p_co = np.abs(np.diff(hp))
                p_co = np.where(p_co < 5e-3, 0, p_co)
                n_co = p_co.sum()
                if self.ploidy_type.startswith('diploid'):
                    n_co *= 2
                cb_info.append(f'{chrom}: {n_co:.2f}')
            cb_info = ', '.join(cb_info)
            rows.append(f"<tr><td>{cb}</td><td>{cb_info}</td></tr>")
        return ''.join(rows), 'Estimated crossovers'
