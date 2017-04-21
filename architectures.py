def mlp(state_size, action_size):
	from keras.models import Sequential
	from keras.layers import Dense, Flatten, Input

	model = Sequential()
	model.add(Dense(32, activation='relu', input_shape=(state_size,)))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(action_size))
	model.compile(loss='mse', optimizer='RMSprop', metrics=['mae'])
	model.summary()

	return model 


def dueling_network_naive(state_size, action_size):
	from keras.models import Model
	from keras.layers import Dense, Flatten, Input, RepeatVector, add

	inputs = Input(shape=(state_size,))
	dense1 = Dense(32, activation='relu')(inputs)
	dense2 = Dense(32, activation='relu')(dense1)
	state_value = Dense(1)(dense2)
	advantage = Dense(action_size)(dense2)
	rep_state_value = Flatten()(RepeatVector(action_size)(state_value))
	action_value = add([rep_state_value, advantage])
	model = Model(inputs=inputs, outputs=action_value)
	model.compile(optimizer='RMSprop', loss='MSE', metrics=['mae'])
	model.summary()

	return model 


def dueling_network_max(state_size, action_size):
	from keras.models import Model
	from keras.layers import Dense, Flatten, Input, RepeatVector
	from keras.layers import add, Lambda, Permute, GlobalMaxPooling1D
	import keras.backend as K 

	inputs = Input(shape=(state_size,))
	dense1 = Dense(32, activation='relu')(inputs)
	dense2 = Dense(32, activation='relu')(dense1)
	state_value = Dense(1)(dense2)
	advantage = Dense(action_size)(dense2)
	max_advantage = GlobalMaxPooling1D()(Permute((2, 1))(RepeatVector(1)(advantage)))
	rep_max_advantage = Flatten()(RepeatVector(action_size)(max_advantage))
	rep_state_value = Flatten()(RepeatVector(action_size)(state_value))
	action_value = add([rep_state_value, advantage, rep_max_advantage])
	model = Model(inputs=inputs, outputs=action_value)
	model.compile(optimizer='RMSprop', loss='MSE', metrics=['mae'])
	model.summary()

	return model 


def dueling_network_ave(state_size, action_size):
	from keras.models import Model
	from keras.layers import Dense, Flatten, Input, RepeatVector
	from keras.layers import add, Lambda, Permute, GlobalAveragePooling1D
	import keras.backend as K 

	inputs = Input(shape=(state_size,))
	dense1 = Dense(32, activation='relu')(inputs)
	dense2 = Dense(32, activation='relu')(dense1)
	state_value = Dense(1)(dense2)
	advantage = Dense(action_size)(dense2)
	ave_advantage = GlobalAveragePooling1D()(Permute((2, 1))(RepeatVector(1)(advantage)))
	rep_ave_advantage = Flatten()(RepeatVector(action_size)(ave_advantage))
	rep_state_value = Flatten()(RepeatVector(action_size)(state_value))
	action_value = add([rep_state_value, advantage, rep_ave_advantage])
	model = Model(inputs=inputs, outputs=action_value)
	model.compile(optimizer='RMSprop', loss='MSE', metrics=['mae'])
	model.summary()

	return model 