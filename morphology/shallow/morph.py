from morph_NN import NN_test
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table


def forwards_feature_selection():
    hidden_layers = (100, 100, 100)

    cat = "C:/Users/garza/Documents/git/DeepLearn/morphology/Nair_Abraham_cat.fit"
    data = fits.getdata(cat,1)
    all_feature_names = np.array(Table(data).colnames)
    feature_names = all_feature_names[np.where(all_feature_names != 'TType')]
    feature_selections = [feature_names.shape]

    for i in range(5):
        feature_set = np.empty(0)
        FoM = 0

        while feature_set.shape < feature_names.shape:
            potential_features = np.setdiff1d(feature_names, feature_set)
            print(potential_features.shape)
            potential_features_fom = np.array(potential_features.shape)

            # Might be same value repeated, might need to run for loop
            for feature in potential_features:
                potential_features_fom[np.where(potential_features == feature
                    )] = NN_test(np.append(feature_set, feature), hidden_layers)

            best_fom= np.amax(potential_features_fom)

            if best_fom > FoM:
                np.append(feature_set, potential_features[np.where(potential_features_fom==best_fom)])
                FoM = best_fom
            else:
                break
        feature_selections[np.where(feature_names in feature_set)] += 1

    final_features = feature_names[np.where(feature_selections >= 2)]
    return final_features


def parameter_optimize(feature_set, MAX_LAYERS, MAX_LAYER_SIZE):
    hidden_layers = [10]
    for occurence in hidden_layers:
        FoM = 0
        occurence = ()

        while occurence.shape() <= MAX_LAYERS:
            new_layer_fom = [MAX_LAYER_SIZE]
            new_layer_fom = NN_test(feature_set, range(MAX_LAYER_SIZE))

            best_fom = np.amax(new_layer_fom)
            if best_fom > FoM:
                occurence += (new_layer_fom[best_fom],)
                FoM = best_fom
            else:
                break
                
    # Not exactly sure how to take mean of the length of the tuples, maybe this            
    final_hidden_layers = [np.mean(hidden_layers.shape[0], dtype=int)]
    for i in range(10):
        final_hidden_layers[i] = np.mean(hidden_layers[i], dtype=int)
    return final_hidden_layers


def main():
    MAX_LAYERS = 50
    MAX_LAYER_SIZE = 200

    print("Optimizing feature set")
    input("Press enter to continue...");
    optimal_feature_set = forwards_feature_selection()
    print("Optimal set found to be")
    print(optimal_feature_set)
    print("------------------------------------------------------------")

    print("Optimizing ANN paramters")
    input("Press enter to continue...")
    optimal_hidden_layers = parameter_optimize(optimal_feature_set, MAX_LAYERS, MAX_LAYER_SIZE)
    print("Optimal hidden layers found to be")
    print(optimal_hidden_layers)
    print("------------------------------------------------------------")

    print("Optimal figure of merit (Area Under Curve) found to be")
    print(test(optimal_feature_set, optimal_hidden_layers))


if __name__ == "__main__":
    main()
