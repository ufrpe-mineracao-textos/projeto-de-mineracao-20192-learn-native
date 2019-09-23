from util.util import PrepData, get_noise

# --- Tirando referências -----


path = 'datasets/'
prep = PrepData(path)
prep.align_verses()
prep.clean_data(get_noise(), True)


