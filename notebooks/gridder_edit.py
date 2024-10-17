#!/usr/bin/env python
import itertools
import functools
import numpy as np
from numpy.linalg import norm
from scipy.sparse import coo_matrix
import astropy.units as u
import astropy.coordinates as ac
import astropy.wcs

def default_zea_wcs(center, delta_x, delta_y):
    w = astropy.wcs.WCS(naxis=2)
    w.wcs.crval = [center.ra.deg, center.dec.deg]
    w.wcs.cdelt = np.array([delta_x.to(u.deg).value, delta_y.to(u.deg).value])
    w.wcs.crpix = [1.0, 1.0]
    w.wcs.ctype = ['RA---ZEA', 'DEC--ZEA']
    return w

def query_pix(pm, pix):
    n=len(pm)
    if pix in pm:
        return pm[pix]
    else:
        pm[pix]=n
        return n

def auto_fov_center(sky_coord_list):
    ra=[s.ra for s in sky_coord_list]
    dec=[s.dec.deg for s in sky_coord_list]

    dec_mean=np.mean(dec)*u.deg
    x,y=np.mean(np.array([np.array([np.cos(a.rad), np.sin(a.rad)]) for a in ra]), axis=0)
    ra_mean=np.arctan2(y,x)*u.rad
    return ac.SkyCoord(ra=ra_mean, dec=dec_mean)

def define_pixel(sky_point_list, dx=None, dy=None, w=None):
    assert(dx is not None or dx is not None or w is not None)
    sky_point_flatten=np.array(list(itertools.chain(*sky_point_list)))
    sky_center=auto_fov_center(sky_point_flatten)
    if w is None:
        if dy is None and dx is not None:
            dy=dx
        if dx is None and dy is not None:
            dx = dy
        w=default_zea_wcs(sky_center, dx, dy)
    g=Gridder(w)
    return g.get_ptr_matrix(sky_point_list)

class Gridder:
    def __init__(self, wcs):
        self.wcs=wcs
        self.dx, self.dy=wcs.wcs.cdelt


    def proj(self, sky_coord):
        deg=np.array([[sky_coord.ra.deg, sky_coord.dec.deg]])
        p=self.wcs.wcs_world2pix(deg, 0)
        return (p[0,0], p[0,1])

    def deproj(self, xy):
        p=np.array([[xy[0], xy[1]]])
        s=self.wcs.wcs_pix2world(p, 0)
        return ac.SkyCoord(ra=s[0,0]*u.deg, dec=s[0,0]*u.deg)

    def grid_index(self, sky_coord):
        p = self.proj(sky_coord)
        x, y = p
        i = int(np.round(x ))
        j = int(np.round(y ))
        return (i, j)

    def get_ptr_matrix(self, points):
        pix_map = {}
        result=[]
        ilist = []
        jlist = []
        values = []
        for i, points1 in enumerate(points):
            ilist.append([])
            jlist.append([])
            values.append([])
            for (j,point) in enumerate(points1):
                pix=self.grid_index(point)
                n=query_pix(pix_map, pix)
                ilist[-1].append(j)
                jlist[-1].append(n)
                values[-1].append(1.0)
        npixels=len(pix_map)
        for il, jl, v in zip(ilist, jlist, values):
            result.append(coo_matrix((v, (il, jl)), shape=(len(il), npixels)))
        pixel_idx=np.array([[k[0], k[1]] for k in pix_map])
        return result, pixel_idx

def fill_map(values, pixel_list, pm_list=None, th=None):
    i_min=int(np.min(pixel_list[:,0]))
    i_max=int(np.max(pixel_list[:,0]))
    j_min=int(np.min(pixel_list[:,1]))
    j_max=int(np.max(pixel_list[:,1]))

    image=np.zeros([i_max-i_min+1, j_max-j_min+1])*np.NaN

    if pm_list is None:
        for (i,j,v) in zip(pixel_list[:,0], pixel_list[:,1], values):
            image[i-i_min,j-j_min]=v
    else:
        h=functools.reduce(lambda x,y:x+y, map(lambda x: np.sum(x, axis=0),pm_list))
        h=np.array(h).squeeze()
        for (i, j, v, c) in zip(pixel_list[:,0], pixel_list[:, 1], values, h):
            if c > th:
                image[i-i_min, j-j_min]=v
    return image
    