from scraping.char_dict import get_size

char_emb_training_specs = {
      'MAX_LENGTH' : 256,
      'MIN_LENGTH' : 32,
      'NUM_CLASSES' : get_size(),
    }
