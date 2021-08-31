from g2p_en.g2p import G2p


def main():

  texts = ["I have $250 in my pocket.",  # number -> spell-out
           "popular pets, e.g. cats and dogs",  # e.g. -> for example
           "I refuse to collect the refuse around here.",  # homograph
           "I'm an activationist."]  # newly coined word
  texts = ["for each 'activationist!"]
  texts = ["Suckin' -- I mean helpin' activationist people an' fightin' an' all that."]
  g2p = G2p()
  for text in texts:
    for word in text.split(" "):
      out2 = g2p.predict(word)
      print(word, " ".join(out2))
    #out = g2p(text)
    # print(out)


if __name__ == '__main__':
  main()
