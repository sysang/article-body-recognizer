from article_body_recognizer import char_dict

char_emb_training_specs = {
      'MAX_LENGTH' : 256,
      'MIN_LENGTH' : 32,
      'NUM_CLASSES' : char_dict.get_size(),
    }
