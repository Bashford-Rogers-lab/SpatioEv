/*
 * Run in QuPath after opening:
 * /Volumes/Shihong_3/ST_Pat4_51/background/ST_Pat4_51_backsub.ome.zarr
 *
 * Automate > Show script editor, paste/run this script.
 */

setChannelNames(
    'DNA_1',
    'FMRP',
    'AMH',
    'GFRA1',
    'SOX9',
    'CREM',
    'MAGEA4',
    'INSL3',
    'KRT18',
    'PIWIL4',
    'NaATPase',
    'UTF1',
    'panCadherin',
    'KI67',
    'CLDN11',
    'DMRT1',
    'CX43',
    'STAR',
    'CYP17A1',
    'SYCP3',
    'CYP11A1',
    'PCNA',
    'CD45',
    'CD68',
    'NF2F2',
    'CD31',
    'UTF1_1',
    'Vimentin',
    'VIMENTIN_1',
    'CX43_1',
    'Acrosin',
    'MYH11',
    'SMA',
    'CD8',
    'DNA_25',
    'CLDN11_2'
)

println 'Updated channel names for ' + getCurrentImageName()
