# OLA (https://yjw.info)

- onlinereg for online regression learning.
- pal for passive-aggressive learning.
- pwea for prediction with expert advice.
- cilm for conformal intervals learning machine.
  
## PAL

To run the code:
- download `pal` 
- execute `python paexp.py`
- record `python python paexp.py --record 1`

Changing following few lines will allow the user to have any data, but the please make sure the format is the same.



    for i in range(len(train_dataset.y)):

        fig_left.scatter(x=train_dataset.dataset.x1[i], y=train_dataset.dataset.x2[i], color=cm.cool(train_dataset.dataset.label[i]), alpha=0.5)

        model_PA.fit(train_dataset.feature_vec[i], train_dataset.y[i])
        model_PA_one.fit(train_dataset.feature_vec[i], train_dataset.y[i])

        accuracies_PA.append(test_dataset.valid_training_result(model_PA))
        accuracies_PA_one.append(test_dataset.valid_training_result(model_PA_one))


## CILM

To run the code:
- download `cilm`
- exection example `python devon_rainfall.py`

Changing following lines will allow the user to have any data, but the please make sure the format is the same.



    pd.read_csv("C:\\Users\\lenovo\\OneDrive\\Documents\\academia\\extremexp\\cilm\\data\\devon_rainfall.csv")
    
## PWEA

To run the code:
- download `pwea`
- execution example `seaa.py`

Changing following few lines will allow the user to have any data, but the please make sure the format is the same.

    parser.add_argument("--data-file", default='data/gas.txt', type=str, help="path of the data file")

    array = np.loadtxt(args.data_file, delimiter=' ')
    n = len(array[0][:-1])
    model = SEAA(args.min_val,args.max_val,args.tuning_parameter,args.switch_rate, n,args.a_a)
    for i, a in enumerate(array):
        x, y = a[:-1], a[-1]
        new_y = model.predict(x)
        

