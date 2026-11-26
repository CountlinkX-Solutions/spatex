# John Snow's cholera map, London 1854

A digitization of John Snow's map of the 1854 cholera outbreak in Soho, London: the addresses
with deaths (`SohoPeople`, with `Count` deaths per address), the 13 public water pumps
(`SohoWater`) and the street network (`Soho_Network`). Distributed with libpysal
(`libpysal.examples`, dataset `snow_maps`, BSD-3-Clause).

Source: Snow, J. (1855). *On the Mode of Communication of Cholera*, 2nd edition. London: John
Churchill.

All layers are in Web Mercator (EPSG:3857, `.prj` included). Used by `notebooks/snow_case.livemd`
and `notebooks/es/caso_colera_snow.livemd`.
