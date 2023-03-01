
class DefaultConfig(object):

    NUM_LABELS = 2

    # Training process
    TRAIN_LR = 1e-3
    DROP_OUT_PROB = 0.1

    EMBEDDING_DIMENSION = 100

    print_every = 100




cfg = DefaultConfig()

if __name__ == "__main__":
    a = "Apigenin suppresses cancer cell growth through ERbeta"
    #print(a[47:50])
    print(a[20:38])

