import sys
from datetime import datetime

import galsim
import numpy as np
import pytz
from astropy.io import fits
from roman_imsim.utils import roman_utils
from scipy.ndimage import gaussian_filter



def get_psf_fits(obsid, outdir, oversample_factor = 8, sed_type = 'flat' , stamp_size = 512, sed = None, normalize = True):

    
    st_model = galsim.DeltaFunction()

    if sed_type == 'flat':
        st_model = st_model*galsim.SED(lambda x:1, 'nm', 'fphotons')
    if sed_type == 'lin':
        st_model = st_model*galsim.SED(lambda x:x, 'nm', 'fphotons')
    if sed_type == 'quad':
        st_model = st_model*galsim.SED(lambda x: x**2, 'nm', 'fphotons')
    if sed_type == 'real':
        assert (sed is not None)
        st_model = st_model*sed

    
    # put some information in the Primary HDU
    mainhdu = fits.PrimaryHDU()
    mainhdu.header['CFORMAT'] = 'Legendre basis'
    mainhdu.header['PORDER'] = (1, 'bivariate polynomial order')
    mainhdu.header['ABSCISSA'] =  ('u=(x-2044.5)/2044, v=(y-2044.5)/2044', 'x,y start at 1')
    mainhdu.header['NCOEF'] = (4, '(PORDER+1)**2')
    mainhdu.header['SEQ'] = 'for n=0..PORDER { for m=0..PORDER { coef P_m(u) P_n(v) }}'
    mainhdu.header['OBSID'] = obsid
    mainhdu.header['NSCA'] = 18
    mainhdu.header['OVSAMP'] = 8
    mainhdu.header['SIMRUN'] = 'Theta -> OSC'
    hdulist = [mainhdu]

    # make each layer
    for sca in range(1,19):
        util = roman_utils('was.yaml',image_name='Roman_WAS_simple_model_Y106_{:d}_{:d}.fits.gz'.format(obsid,sca))
        if normalize:
            st_model = st_model.withFlux(1.,util.bpass)
    
        out_psf = np.zeros((6,stamp_size,stamp_size))
        x_ = [   1.,4088.,   1.,4088.,2044.5, 500.]
        y_ = [   1.,   1.,4088.,4088.,2044.5,1000.]
        for j in range(5):
            x_[j] = .9*x_[j]+.1*2044.5
            y_[j] = .9*y_[j]+.1*2044.5
            psf = util.getPSF(x_[j],y_[j],8)
            psf_image = galsim.Convolve(st_model, galsim.Transform(psf,jac=8*np.identity(2)))
            stamp = galsim.Image(stamp_size,stamp_size,wcs=util.wcs)
            arr = psf_image.drawImage(util.bpass,image=stamp,wcs=util.wcs,method='no_pixel')
            out_psf[j,:,:] = arr.array
                    
        out_psf_coef = np.zeros((5,stamp_size,stamp_size))
        out_psf_coef[0,:,:] = out_psf[4,:,:]
        out_psf_coef[1,:,:] = (out_psf[1,:,:] + out_psf[3,:,:] - out_psf[0,:,:] - out_psf[2,:,:])/2./(4087*.9) * 2044
        out_psf_coef[2,:,:] = (out_psf[2,:,:] + out_psf[3,:,:] - out_psf[0,:,:] - out_psf[1,:,:])/2./(4087*.9) * 2044
        out_psf_coef[3,:,:] = (out_psf[0,:,:] + out_psf[3,:,:] - out_psf[1,:,:] - out_psf[2,:,:])/4./(4087*.45)**2 * 2044**2

        # convolution
        sig_mtf = 0.3279*8 # 8 for oversampled pixels
        for j in range(4):
            out_psf_coef[j,:,:] = 0.17519*gaussian_filter(out_psf_coef[j,:,:], 0.4522*sig_mtf, truncate=7.0)\
                                 +0.53146*gaussian_filter(out_psf_coef[j,:,:], 0.8050*sig_mtf, truncate=7.0)\
                                 +0.29335*gaussian_filter(out_psf_coef[j,:,:], 1.4329*sig_mtf, truncate=7.0)

        
        hdu = fits.ImageHDU(out_psf_coef[:4,128:-128,128:-128].astype(np.float32))
        hdu.header['OBSID'] = obsid
        hdu.header['SCA'] = sca
        
        hdulist.append(hdu)
    fits.HDUList(hdulist).writeto(outdir +'/psf_polyfit_{:d}.fits'.format(obsid), overwrite=True)


# parameters
sed_type = 'flat'
sed = None
eff_const = False
normalize = True
outdir = './input_psf/psf_flatsed'
obsid = int(sys.argv[1])
stamp_size = 512

## Run psf generation
get_psf_fits(obsid, outdir, sed_type=sed_type, sed =sed, normalize = normalize, eff_const = eff_const)
