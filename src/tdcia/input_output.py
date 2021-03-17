import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class input_data:
    """Data object that handles reading in the data.

    Args:
        filename (string): The filename of the input file.
        filedir (string): The directory containing the input file.

    """

    def __init__(self, filename, filedir):
        self.name = filename
        self.dir = filedir
        self.df = {'expec.t', 'npop.t'}
        self.np = {'table.dat', 'efield.t', 'nstate_i.t'}
        self.read_df = self.check_read()

    def check_read(self):
        """Method that determines if numpy array or pandas dataframe\
        are to be read, based on the input file name.

        Returns:
            bool: True if pandas dataframe is to be read; false if numpy\
            array is to be read."""
        if self.name in self.df:
            read_df = True
            print('Reading pandas dataframe')
        elif self.name in self.np:
            read_df = False
            print('Reading numpy array')
        else:
            print('Error: This file type is unknown')
            print('Please provide one of the following:')
            print(self.df, self.np)
            exit()
        return read_df

    def read_in(self):
        """Reads in the data into an numpy array or pandas dataframe.

        Returns:
            numpy array/pandas dataframe: Returns the data either in an array\
            or dataframe."""
        if self.read_df:
            # method to read in the data files as a dataframe
            name = '{}{}'.format(self.dir, self.name)
            print('Reading from file {} - pandas'.format(name))
            self.data = pd.read_csv(name, r'\s+')
        else:
            # method to read in the data files as numpy arrays
            name = '{}{}'.format(self.dir, self.name)
            print('Reading from file {} - numpy'.format(name))
            self.data = np.loadtxt(name, skiprows=1)
            self.data = self.data.T
        return self.data


class output_data:
    """Data object that handles plotting and writing data.

    Args:
        data (numpy array/pandas dataframe): The data to be plotted/printed.
        indices (integer list): The column indices that remain if \
        insignificant entries were skipped.
        option (string): The option that handles the output processing -\
        choose from: `expecval`, `MOpop`, `transdipmom`, `efield`, `aucofu`.
        data2 (numpy array/pandas dataframe): The additional data, if two sets\
        of data are processed for that specific output option.
        """

    def __init__(self, data, indices, outdir, option, data2=0):
        self.data = data
        self.index = indices
        self.data2 = data2  # optional
        self.option = option
        self.outdir = outdir  # output directory for files
        # case switch option in python using a dictionary and function names
        self.objects = {'expecval': self.plot0, 'MOpop': self.plot1,
                        'transdipmom': self.plot2,
                        'efield': self.plot3, 'aucofu': self.plot4}
        self.case = self.objects.get(self.option, self.plot5)
        # call the selected function
        self.case()

    # plotting options
    def plotparams(self, myplot, strx, stry):
        """Method to define plotting parameters such as font size.

        Args:
            myplot (string): Name of the figure object (typically 'ax').
            strx (string): Name of the x axis label.
            stry (string): Name of the y axis label."""
        myfont = 18
        myplot.xaxis.set_tick_params(labelsize=myfont)
        myplot.yaxis.set_tick_params(labelsize=myfont)
        myplot.set_xlabel(strx, fontsize=myfont)
        myplot.set_ylabel(stry, fontsize=myfont)
        myplot.legend(loc='upper right', shadow=False,
                      fontsize=myfont - 4, borderpad=0.1,
                      labelspacing=0, handlelength=1)

    def plot0(self):
        """Expectation value output method.
        Generates plot - the specified output data is generated and saved to\
        the corresponding file."""
        print('Plotting expectation values')
        for i in range(1, len(self.data.columns)):
            print(self.index[i])
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(self.data[self.index[0]], self.data[self.index[i]],
                    label='{}'.format(self.index[i]))
            self.plotparams(ax, "time (fs)", "expectation value (au)")
            plt.savefig('{}/Figure_expec_{}.pdf'.format(self.outdir,
                        self.index[i]),
                        dpi=300, bbox_inches='tight')

    def plot1(self):
        """MO populations output method.
        Generates data files and plot - the specified output data is generated\
        and saved to the corresponding files."""
        print('Plotting MO populations and correlation')
        sns.pairplot(self.data, kind='kde', corner=True)
        plt.savefig('{}/Figure_npopcorr.pdf'.format(self.outdir),
                    dpi=300, bbox_inches='tight')
        print('Writing correlation matrix')
        self.data2.to_csv('{}/corrmat.dat'.format(self.outdir), header=True,
                          sep=' ')

    def plot2(self):
        """Transition dipole moment output method.
        Generates data files and plot - the specified output data is generated\
        and saved to the corresponding files."""
        print('Writing L2 norm of transition dipole moments - ')
        np.savetxt('{}/l2norm.dat'.format(self.outdir),
                   np.column_stack((self.data[0], self.data[1], self.data[2])),
                   newline='\n', header='x   y   z')
        print('Plotting L2 norm')
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(0, len(self.data))
        plt.bar(x, self.data)
        plt.xticks(x, ('x', 'y', 'z'))
        self.plotparams(ax, "vector component", "L2 norm")
        plt.savefig('{}/Figure_l2norm.pdf'.format(self.outdir),
                    dpi=300, bbox_inches='tight')

    def plot3(self):
        """Electric field output method.
        Generates data files and plot - the specified output data is generated\
        and saved to the corresponding files."""
        print('Plotting FT of electric field')
        labels = ['time', 'x', 'y', 'z']
        for i in range(1, len(self.data)):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(np.real(self.data[0]), np.real(self.data[i]),
                    label="real part {}".format(labels[self.index[0][i]]))
            ax.plot(np.real(self.data[0]), np.imag(self.data[i]),
                    label="imaginary part {}".format(labels[self.index[0][i]]))
            ax.plot(np.real(self.data[0]), np.abs(self.data[i]),
                    label="absolute value {}".format(labels[self.index[0][i]]))
            self.plotparams(ax, "time (fs)", "Fourier transform")
            plt.savefig('{}/Figure_efield_FT_{}.pdf'.
                        format(self.outdir, labels[self.index[0][i]]),
                        dpi=300, bbox_inches='tight')

    def plot4(self):
        """Autocorrelation function output method.
        Generates data files and plot - the specified output data is generated\
        and saved to the corresponding files."""
        print('Writing autocorrelation function')
        np.savetxt('{}/aucofu.t'.format(self.outdir), np.column_stack([
                   np.real(self.data[0]),
                   np.real(self.data[1]), np.imag(self.data[1]),
                   np.abs(self.data[1])]), newline='\n',
                   header='time (fs)   re(aucofu)   imag(aucof)   abs(aucofu)')
        print('Plotting autocorrelation function')
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(np.real(self.data[0]), np.real(self.data[1]),
                label="real part")
        ax.plot(np.real(self.data[0]), np.imag(self.data[1]),
                label="imaginary part")
        ax.plot(np.real(self.data[0]), np.abs(self.data[1]),
                label="absolute value")
        self.plotparams(ax, "time (fs)", "autocorrelation function")
        plt.savefig('{}/Figure_autocorrelation_function.pdf'.format(
                    self.outdir), dpi=300, bbox_inches='tight')
        print('Plotting FT of autocorrelation function')
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(np.real(self.data2[0]), np.real(self.data2[1]),
                label="real part")
        ax.plot(np.real(self.data2[0]), np.imag(self.data2[1]),
                label="imaginary part")
        ax.plot(np.real(self.data2[0]), np.abs(self.data2[1]),
                label="absolute value")
        self.plotparams(ax, "energy (au)", "Fourier transform")
        plt.savefig('{}/Figure_autocorrelation_function_FT.pdf'.format(
            self.outdir), dpi=300, bbox_inches='tight')

    def plot5(self):
        """Default output method - no output."""
        print('Error: Method not found')
        print('Select from the following:', self.objects)
        exit()

# input_data('expec.t', 'data/')
# input_data('efield.t', 'data/')
# myobjin = input_data('nstate_i.t', 'data/')
# wavef = myobjin.read_in()
# time, aucofu = fl.aucofu(wavef)
# myobjout = output_data(np.vstack((time, aucofu)), option='aucofu')
