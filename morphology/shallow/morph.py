from morph_NN import NN_test
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from time import sleep
from math import floor


def forwards_feature_selection():
    print("Optimizing feature set")
    input("Press enter to continue...")

    hidden_layers = (192, 12, 181)

    
    # Generate numerical dataset
    cat = "/home/ben/Documents/git/DeepLearn/morphology/Nair_Abraham_cat.fit"
    data = fits.getdata(cat,1)
    all_feature_names = np.array(Table(data).colnames)
    all_feature_names_set = set(all_feature_names)
    feature_names = all_feature_names[np.where(all_feature_names!="TType")]
    feature_selections = np.zeros(feature_names.shape)

    for i in range(5):
        print("Iteration", (i+1), "/ 5")
        feature_set = np.empty(0)
        FoM = 0

        while feature_set.shape < feature_names.shape:
            potential_features = np.setdiff1d(feature_names, feature_set)
            potential_features_fom = []

            for feature in potential_features:
                print("Testing", feature)
                print("Testing Feature Set", np.append(feature_set, feature))
                potential_features_fom.append(NN_test(np.append(feature_set, feature), hidden_layers))

            best_fom= np.amax(potential_features_fom)

            if best_fom > FoM:
                best_potential_feature = potential_features[np.where(potential_features_fom==best_fom)]
                feature_set = np.append(feature_set, best_potential_feature)
                FoM = best_fom
                print(best_potential_feature, "gave highest FoM of", best_fom, ", adding...")
                sleep(3)
            else:
                print("Finished iteration")
                print("Features selected:", feature_set)
                feature_selections[np.where(np.isin(all_feature_names, feature_set))] += 1
                break

    final_features = feature_names[np.where(feature_selections >= 2)]
    return final_features, feature_selections, feature_names


def parameter_optimize(feature_set, MAX_LAYERS, MAX_LAYER_SIZE):
    print("Optimizing ANN paramters")
    input("Press enter to continue...")

    hidden_layers_iterations = []
    for i in range(10):
        print("Iteration", (i+1), "/ 10")
        FoM = 0
        occurence = []

        while len(occurence) <= MAX_LAYERS:
            new_layer_fom = []
            for j in range(1, MAX_LAYER_SIZE+1):
                print(j)
                print("Testing hidden layers", occurence + [j])
                new_layer_fom.append(NN_test(feature_set, tuple(occurence + [j])))

            best_fom = np.amax(new_layer_fom)
            if best_fom > FoM:
                layer_size = 1+np.where(new_layer_fom==best_fom)[0][0]
                occurence.append(layer_size)
                FoM = best_fom
                print("Best layer with FoM", best_fom, "has size", layer_size)
            else:
                print("Finished iteration")
                hidden_layers_iterations.append(occurence)
                break
    

    print("Finished testing parameters")
       
    final_hidden_layers = []

    final_num_layers = 0
    for iteration in hidden_layers_iterations:
        final_num_layers += len(iteration)
    final_num_layers = floor(final_num_layers/10)

    for i in range(final_num_layers):
        final_hidden_layers.append(np.mean(hidden_layers_iterations[:,i], dtype=int))

    return zip(final_hidden_layers, hidden_layers_iterations)


def main():
    MAX_LAYERS = 8
    MAX_LAYER_SIZE = 200


    f = open("morph.txt", "w+")

    '''

    optimal_feature_set, feature_selections, feature_names = forwards_feature_selection()

    print("Optimal set found to be:\n")
    print(optimal_feature_set)

    f.write("Optimal feature set:\n")
    f.write(str(optimal_feature_set))
    f.write("\nFeature selections:\n")
    for name, selection in zip(feature_names, feature_selections):
        f.write(str(name))
        f.write(", ")
        f.write(str(selection))
        f.write("\n")

    print("------------------------------------------------------------")
    '''

    optimal_feature_set=['g-r', 'bOverA', 'Rp50_g', 'Rp90_g', 'Bar', 'tails', 'Ring_flag', 'V/Vmax']

    '''
    optimal_hidden_layers, iterations = parameter_optimize(optimal_feature_set, MAX_LAYERS, MAX_LAYER_SIZE)
    
    print("Optimal hidden layers found to be")
    print(optimal_hidden_layers)

    f.write("\nOptimal hidden layers:\n")
    f.write(str(optimal_hidden_layers))
    f.write("Hidden layer iterations:\n")
    for i in range(10):
        f.write(str(iterations[i]))
        f.write("\n")    

    print("------------------------------------------------------------")
    '''

    optimal_hidden_layers = [192, 12, 181]

    AUC_optimized = NN_test(optimal_feature_set, tuple(optimal_hidden_layers))

    print("Optimal figure of merit (Area Under Curve) found to be", AUC_optimized)
    
    f.write("Optimized figure of merit:")
    f.write(str(AUC_optimized))

    f.close()


if __name__ == "__main__":
    main()
