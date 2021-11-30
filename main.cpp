#include<stdio.h>
#include <stdlib.h>
#include<math.h>
#include <time.h>

#define BOARD_SIZE 8 //”Õ–Ê‚Ìˆê•Ó‚Ì”
#define EPISODE 15000 //ƒGƒsƒ\[ƒh”
#define MEMORY_SIZE 800 //ˆê“x‚É•Û‘¶‚·‚éExperience Reply‚Ì”
#define BATCH_SIZE 400 //ƒoƒbƒ`ƒTƒCƒY
#define EPISODE_INTERVAL 80 //ŠwK‚ðs‚¤•p“x

#define INPUT_DIM 5 //“ü—ÍŽŸŒ³ió‘Ô”j
#define MIDDLE_DIM 16 //‰B‚ê‘w‚ÌŽŸŒ³
#define OUTPUT_DIM 64 //o—ÍŽŸŒ³

typedef struct {
	int x;
	int y;
} coordinate;

//éŒ¾Žž‚É‰Šú‰»–Y‚ê‚È‚¢(0‚Å‰Šú‰»)
typedef struct {
	int enable[64]; //1‚¾‚Æ’u‚¯‚é
	int count;
} enable_put;

typedef struct {
	int index;
	float value;
} array_with_index;

typedef struct {
	int state[INPUT_DIM];
	int action;
	int next_state[INPUT_DIM];
	float reward;
} experience_reply;

//is•ûŒü
const coordinate direction[8]{
	{1,-1},
	{1,0},
	{1,1},
	{0,1},
	{-1,1},
	{-1,0},
	{-1,-1},
	{0,-1},
};

//	q’l(~‡)
int cmpDescValue(const void* n1, const void* n2)
{
	if (((array_with_index*)n1)->value < ((array_with_index*)n2)->value)
	{
		return 1;
	}
	else if (((array_with_index*)n1)->value > ((array_with_index*)n2)->value)
	{
		return -1;
	}
	else
	{
		return 0;
	}
}

coordinate transAddressToCoordinate(int address) {
	coordinate tmp;
	tmp.x = address % BOARD_SIZE;
	tmp.y = address / BOARD_SIZE;

	return tmp;
}

int transCoordinateToAddress(coordinate coor) {
	return coor.y * BOARD_SIZE + coor.x;
}

coordinate addCoordinate(coordinate a, coordinate b) {
	coordinate c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;

	return c;
}

coordinate multiCoordinateAndScala(coordinate a, int scala) {
	coordinate b;
	b.x = a.x * scala;
	b.y = a.y * scala;

	return b;
}

void setIndex(array_with_index* src, float *value, int count) {
	for (int i = 0; i < count; i++) {
		src[i].value = value[i];
		src[i].index = i;
	}
}

void copyIntArray(int* src, int* dst, int count) {
	for (int i = 0; i < count; i++) dst[i] = src[i];
}

void copyFloatArray(float* src, float* dst, int count) {
	for (int i = 0; i < count; i++) dst[i] = src[i];
}

float setNormalDistribution(float x, float mean, float sigma) {
	return 1 / sqrtf(2 * 3.14 * sigma) * expf(-powf(x - mean, 2) / (2 * sigma));
}

void setUniformDistributionToArray(float* output, int count, int input_count) {
	srand((unsigned int)time(NULL));

	for (int i = 0; i < count; i++) output[i] = setNormalDistribution((float)((rand() + 0.5) / (RAND_MAX + 1)), 0, input_count);
}

int setRandomIndex(int min, int max, int seed) {
	srand((unsigned int)time(NULL) + seed);

	return (rand() % (max - min + 1)) + min;
}

void setUniqueIndexArray(int* output, int output_count, int count) {
	srand((unsigned int)time(NULL) + (unsigned int)rand());
	int* flg, done = 0;
	flg = (int*)malloc(output_count * sizeof(int));
	for (int i = 0; i < output_count; i++) flg[i] = (int)0;
	while (done <= count) {
		const int tmp = rand() % output_count;
		if (flg[tmp] == 0) {
			output[done] = tmp;
			flg[tmp] = 1;
			done++;
		}
	}
	free(flg);
}

//ƒ‰ƒ“ƒ_ƒ€s“®:2AQ’l‚©‚ç1
int selectEpisilonOrGreedy(float epsilon_start, float epsilon_end, float epsilon_decay, int episode) {
	const float threshold = epsilon_end + (epsilon_start - epsilon_end) * expf(-episode / epsilon_decay);
	srand((unsigned int)time(NULL) + (unsigned int)episode * 100);

	return (rand() + 0.5) / (RAND_MAX + 1) < threshold ? 2 : 1;
}

enable_put checkPutCapability(int* board, int current_color) {
	enable_put check = { {0}, 0 };
	for (int address = 0; address < BOARD_SIZE * BOARD_SIZE; address++) {
		if (board[address] != 0) continue;
		for (int dir = 0; dir < 8; dir++) {
			for (int dir_value = 1; dir_value <= 6; dir_value++) {
				const coordinate current_coor = addCoordinate(transAddressToCoordinate(address), multiCoordinateAndScala(direction[dir], dir_value));

				if (current_coor.x < 0 || current_coor.x >= BOARD_SIZE || current_coor.y < 0 || current_coor.y >= BOARD_SIZE) break;
				if (dir_value == 1 && board[transCoordinateToAddress(current_coor)] != current_color * -1) break;
				if (dir_value != 1 && board[transCoordinateToAddress(current_coor)] == 0) break;

				if (dir_value != 1 && board[transCoordinateToAddress(current_coor)] == current_color) {
					check.enable[address] = 1;
					check.count++;
					break;
				}
			}
			if (check.enable[address] == 1) break;
		}
	}

	return check;
}

experience_reply createRecoed(int* state, int action, int* next_state, float reward) {
	experience_reply a;
	for (int i = 0; i < INPUT_DIM; i++) {
		a.state[i] = state[i];
		a.next_state[i] = next_state[i];
	}
	a.action = action;
	a.reward = reward;

	return a;
}

void putBoard(int* board, int number, int current_color) {
	board[number] = current_color;
	float reward = 0;
	for (int dir = 0; dir < 8; dir++) {
		int count = 0;
		for (int dir_value = 1; dir_value <= BOARD_SIZE - 2; dir_value++) {
			const coordinate current_coor = addCoordinate(transAddressToCoordinate(number), multiCoordinateAndScala(direction[dir], dir_value));

			if (current_coor.x < 0 || current_coor.x >= BOARD_SIZE || current_coor.y < 0 || current_coor.y >= BOARD_SIZE) break;
			if (dir_value == 1 && board[transCoordinateToAddress(current_coor)] != current_color * -1) break;
			else if (dir_value == 1) count++;

			if (dir_value > 1) {
				if (board[transCoordinateToAddress(current_coor)] == 0) break;
				else if (board[transCoordinateToAddress(current_coor)] == current_color) {
					for (int i = 1; i <= count; i++) {
						const coordinate now_coor = addCoordinate(transAddressToCoordinate(number), multiCoordinateAndScala(direction[dir], i));
						board[transCoordinateToAddress(now_coor)] = current_color;
					}
					break;
				}
				else count++;
			}
		}
	}
}

/*
* state[0]FŽ©•ª‚ÌŠp‚ÌÎ‚Ì”
* state[1]F‘ŠŽè‚ÌŠp‚ÌÎ‚Ì”
* state[2]FŽ©•ª‚ÌÎ‚Ì”-‘ŠŽè‚ÌÎ‚Ì”
* state[3]FÎ‚Ì’u‚©‚ê‚Ä‚¢‚È‚¢êŠ‚Ì”
* state[4]F•s—˜‚É‚È‚éêŠ‚É’u‚©‚ê‚Ä‚¢‚éÎ‚Ì”FŽ©•ª-‘ŠŽè
*/
void createState(int* board, int* state) {
	for (int i = 0; i < INPUT_DIM;i++) state[i] = 0;

	for (int i = 0;i < BOARD_SIZE * BOARD_SIZE; i++) {
		if (i == 0 || i == 7 || i == 56 || i == 63) { //Šp
			if (board[i] == 1) state[0]++;
			else if (board[i] == -1) state[1]++;
		}
		if (board[i] == 0) state[3]++;
		if (i == 1 || i == 8 || i == 9 || i == 6 || i == 14 || i == 15 || i == 48
			|| i == 49 || i == 57 || i == 54 || i == 55 || i == 62) state[4] += board[i];

		state[2] += board[i];
	}
}

float calcReward(int* board, int black_put_number, int white_put_number, int effort) {
	float reward = 0.0;
	int black = 0, white = 0;
	for (int index = 0; index < BOARD_SIZE * BOARD_SIZE; index++) {
		if (board[index] == 1) black++;
		else white++;
	}
	if (effort == BOARD_SIZE * BOARD_SIZE - 4) {
		if (black > white) reward++;
		else if (black <= white) reward--;
	}
	else if(white == 0 || black == 0){
		if (white == 0) reward++;
		if (black == 0) reward--;
	}

	if (black_put_number == 0 || black_put_number == 7 || black_put_number == 56 || black_put_number == 63) reward += 0.2;
	//HACK:Œ©‚Ã‚ç‚¢
	else if (black_put_number == 1 || black_put_number == 8 || black_put_number == 9 || black_put_number == 6 || black_put_number == 14 || black_put_number == 15 || black_put_number == 48
		|| black_put_number == 49 || black_put_number == 57 || black_put_number == 54 || black_put_number == 55 || black_put_number == 62) reward -= 0.1;

	return reward;
}

int choicePutValue(enable_put enable_array, array_with_index* q_value) {
	for (int q_value_index = 0; q_value_index < BOARD_SIZE * BOARD_SIZE; q_value_index++) {
		if (enable_array.enable[q_value[q_value_index].index] == 1) return q_value[q_value_index].index;
	}

	return -1;
}

int choiceRamdomPutValue(enable_put enable_array, int count) {
	int tmp = 0;
	for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
		if (enable_array.enable[i] == 1) {
			tmp++;
			if (tmp == count) return i;
		}
	}
}

/************************

‡“`”ÀŒvŽZ

************************/

//OPTIMIZE:‚‘¬‰»‰Â”\
void calcForwardMiddleClass(int* input, float* output, float* weight, int input_dim, int output_dim) {
	for (int i = 0; i < output_dim; i++) output[i] = 0;

	for (int j = 0; j < input_dim * output_dim; j++) output[j / input_dim] += input[j % input_dim] * weight[j];

	for (int k = 0; k < output_dim; k++) output[k] = tanhf(output[k]);
}

//OPTIMIZE:‚‘¬‰»‰Â”\
void calcForwardFullcombined(float* input, float* output, float* weight, int input_dim, int output_dim) {
	for (int i = 0; i < output_dim; i++) output[i] = 0;

	for (int j = 0; j < input_dim * output_dim; j++) output[j / input_dim] += input[j % input_dim] * weight[j];

	for (int k = 0; k < output_dim; k++) output[k] = tanhf(output[k]);
}

void calcForwardpropagation(int* input, float* output, float* weight_middle, float* weight_full, int input_dim, int middle_dim, int output_dim) {
	float *middle_output;
	middle_output = (float*)malloc(middle_dim * sizeof(float));
	calcForwardMiddleClass(input, middle_output, weight_middle, input_dim, middle_dim);
	calcForwardFullcombined(middle_output, output, weight_full, middle_dim, output_dim);
	free(middle_output);
}

/************************

‡“`”ÀŒvŽZI‚í‚è

************************/

/************************

Œë·‹t“`”À

************************/

void pullOutExperienceMemory(experience_reply* src, experience_reply* dst, int batch_size, int memory_size) {
	int* batch_array;
	batch_array = (int*)malloc(batch_size * sizeof(int));
	setUniqueIndexArray(batch_array, memory_size, batch_size);

	for (int i = 0; i < batch_size; i++) {
		dst[i] = src[batch_array[i]];
	}
	//free(batch_array);
}

void calcForwardpropagationInBackpropagation(int* input, float* output, float* weight_middle, float *middle_output, float* weight_full, int input_dim, int middle_dim, int output_dim) {
	calcForwardMiddleClass(input, middle_output, weight_middle, input_dim, middle_dim);
	calcForwardFullcombined(middle_output, output, weight_full, middle_dim, output_dim);
}

void calcErrorBackPropagation(int *input, float* d3, float* middle_output, float* final_delta, float* middle_delta, int input_dim, int final_dim, int middle_dim) {
	for (int i = 0; i < final_dim * middle_dim; i++) {
		final_delta[i] = d3[i % final_dim] * middle_output[i / middle_dim];
	}

	float tmp_middle[BOARD_SIZE * BOARD_SIZE] = { 0 };
	for (int j = 0; j < middle_dim * final_dim; j++) {
		tmp_middle[j / middle_dim] += d3[j % final_dim] / powf(coshf(atanhf(middle_output[j / middle_dim])), 2);
	}

	for (int j = 0; j < middle_dim * input_dim; j++) {
		middle_delta[j] = tmp_middle[j / middle_dim] * input[j % input_dim];
	}
}

void updateWeight(float* middle_weight, float* final_weight, float* middle_delta, float* final_delta, float epsilon, int input_dim, int middle_dim, int output_dim) {
	for (int i = 0; i < input_dim * middle_dim; i++) {
		middle_weight[i] = middle_weight[i] - epsilon * middle_delta[i] / BATCH_SIZE;
	}

	for (int i = 0; i < output_dim * middle_dim; i++) {
		final_weight[i] = final_weight[i] - epsilon * final_delta[i] / BATCH_SIZE;
		//if (i == output_dim * middle_dim - 1) printf("%lf\n", final_delta[i]);
	}
}
void doTrainQNetwork(experience_reply* reply, float* middle_weight, float* final_weight, int input_dim, int middle_dim, int output_dim) {
	experience_reply batch[BATCH_SIZE];
	float target_middle_weight[MIDDLE_DIM * INPUT_DIM], target_final_weight[MIDDLE_DIM * OUTPUT_DIM];
	float target_q_value[OUTPUT_DIM], q_value[OUTPUT_DIM], diff_q_value[OUTPUT_DIM];
	float middle_output[MIDDLE_DIM];
	float middle_delta[MIDDLE_DIM * INPUT_DIM], final_delta[MIDDLE_DIM * OUTPUT_DIM];
	copyFloatArray(middle_weight, target_middle_weight, MIDDLE_DIM * INPUT_DIM);
	copyFloatArray(final_weight, target_final_weight, MIDDLE_DIM * OUTPUT_DIM);
	pullOutExperienceMemory(reply, batch, BATCH_SIZE, MEMORY_SIZE);

	for (int batch_index = 0; batch_index < BATCH_SIZE; batch_index++) {
		calcForwardpropagation(batch[batch_index].next_state, target_q_value, target_middle_weight, target_final_weight, INPUT_DIM, MIDDLE_DIM, OUTPUT_DIM);
		calcForwardpropagationInBackpropagation(batch[batch_index].state, q_value, middle_weight, middle_output, final_weight, INPUT_DIM, MIDDLE_DIM, OUTPUT_DIM);
		array_with_index q_value_with_index[OUTPUT_DIM];
		setIndex(q_value_with_index, target_q_value, OUTPUT_DIM);
		qsort(q_value_with_index, OUTPUT_DIM, sizeof(array_with_index), cmpDescValue);

		for (int i = 0; i < OUTPUT_DIM; i++) {
			if (i == q_value_with_index[0].index) {
				diff_q_value[i] = q_value[i] - batch[batch_index].reward + 0.99 * q_value_with_index[0].value;
			}
			else
				diff_q_value[i] = q_value[i] - batch[batch_index].reward + 0.99 * q_value_with_index[0].value;//diff_q_value[i] = 0;
		}
		calcErrorBackPropagation(batch[batch_index].state, diff_q_value, middle_output, final_delta, middle_delta, input_dim, output_dim, middle_dim);
		updateWeight(middle_weight, final_weight, middle_delta, final_delta, 0.1, input_dim, middle_dim, output_dim);
	}
}

/************************

Œë·‹t“`”ÀI‚í‚è

************************/

void resetBoard(int* board) {
	for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
		if (i == 27 || i == 36) board[i] = -1;
		else if (i == 28 || i == 35) board[i] = 1;
		else board[i] = 0;
	}
}

void resetEpisode(experience_reply* input) {
	for (int i = 0; i < MEMORY_SIZE; i++) {
		input[i] = { {0},0,{0},0 };
	}
}

void runSetBoard(int* board, array_with_index *q_value_with_index, int current_color, int put_number) {
	putBoard(board, put_number, current_color);
}

void printBoard(int* board) {
	puts(" ‚P‚Q‚R‚S‚T‚U‚V‚W");
	for (int i = 0; i < BOARD_SIZE; i++) {
		printf("%d", i);
		for (int j = 0; j < BOARD_SIZE; j++) {
			if (board[i * BOARD_SIZE + j] == -1) printf("œ");
			else if (board[i * BOARD_SIZE + j] == 1) printf("›");
			else printf("[");
		}
		puts("");
	}
}

void printEnablePut(enable_put enable_array) {
	for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) if(enable_array.enable[i] == 1) printf("%dA", i);
	puts("");
}

void printQValue(array_with_index* index, int count) {
	for (int i = 0;i < count; i++) printf("%dF%lf\n", index[i].index, index[i].value);
}

int main() {
	int board[BOARD_SIZE * BOARD_SIZE] = { 0 }; //‹ó‚«ƒ}ƒX:0A”’:-1A•F1
	experience_reply memory[MEMORY_SIZE];
	float middle_weight[INPUT_DIM * MIDDLE_DIM], combined_weight[MIDDLE_DIM * OUTPUT_DIM];
	setUniformDistributionToArray(middle_weight, INPUT_DIM * MIDDLE_DIM, INPUT_DIM);
	setUniformDistributionToArray(combined_weight, MIDDLE_DIM * OUTPUT_DIM, MIDDLE_DIM);

	int effort = 1, current_color=1, memory_index = 0, pass=0;
	resetBoard(board);
	array_with_index q_value_with_index[OUTPUT_DIM];
	float q_value[OUTPUT_DIM] = { 0 };
	setIndex(q_value_with_index, q_value, OUTPUT_DIM);

	for (int episode = 0; episode < EPISODE; episode++) {
		if (episode % EPISODE_INTERVAL == 0 && episode != 0) {
			doTrainQNetwork(memory, middle_weight, combined_weight, INPUT_DIM, MIDDLE_DIM, OUTPUT_DIM);
			resetEpisode(memory);
		}
		int prev_color = 0, now_color = 0, black_put_value = 999, white_put_value = 999;
		resetBoard(board);
		effort = 1;
		current_color = 1;
		pass = 0;
		while (effort <= 60) {
			int put_value, state[INPUT_DIM], next_state[INPUT_DIM];
			const enable_put enable_array = checkPutCapability(board, current_color);

			if (pass == 2) break;
			if (enable_array.count == 0) {
				current_color *= -1;
				pass++;
				continue;
			}
			else pass = 0;

			createState(board, state); //‚±‚±‚ð‚¢‚¶‚é

			if (current_color == 1) {
				const int action_term = selectEpisilonOrGreedy(0.9, 0.05, 20, episode);
				if (action_term == 2) put_value = choiceRamdomPutValue(enable_array, setRandomIndex(1, enable_array.count, 0));
				else {
					calcForwardpropagation(state, q_value, middle_weight, combined_weight, INPUT_DIM, MIDDLE_DIM, OUTPUT_DIM);
					setIndex(q_value_with_index, q_value, OUTPUT_DIM);
					qsort(q_value_with_index, OUTPUT_DIM, sizeof(array_with_index), cmpDescValue);
					put_value = choicePutValue(enable_array, q_value_with_index);
				}
				black_put_value = put_value;
				prev_color = now_color;
				now_color = current_color;
			}
			else {
				put_value = choiceRamdomPutValue(enable_array, setRandomIndex(1, enable_array.count, 0));
				white_put_value = put_value;
				prev_color = now_color;
				now_color = current_color;
			}
			putBoard(board, put_value, current_color);
			createState(board, next_state);
			if (prev_color == 1 && now_color == -1) {
				memory[memory_index % MEMORY_SIZE] = createRecoed(state, black_put_value, next_state, calcReward(board, black_put_value, white_put_value, effort));
				memory_index++;
			}
			effort++;

			current_color *= -1;
			//printBoard(board);
		}
	}
	int win = 0;

	for (int index = 0;index < 1000; index++) {
		resetBoard(board);
		effort = 1;
		current_color = 1;
		pass = 0;
		while (effort <= 60) {
			if (pass == 2)break;
			int put_value, state[INPUT_DIM];
			const enable_put enable_array = checkPutCapability(board, current_color);

			if (enable_array.count == 0) {
				current_color *= -1;
				pass++;
				continue;
			}
			else pass = 0;

			createState(board, state); //‚±‚±‚ð‚¢‚¶‚é

			if (current_color == 1) {
				calcForwardpropagation(state, q_value, middle_weight, combined_weight, INPUT_DIM, MIDDLE_DIM, OUTPUT_DIM);
				setIndex(q_value_with_index, q_value, OUTPUT_DIM);
				qsort(q_value_with_index, OUTPUT_DIM, sizeof(array_with_index), cmpDescValue);
				put_value = choicePutValue(enable_array, q_value_with_index);
			}
			else {
				put_value = choiceRamdomPutValue(enable_array, setRandomIndex(1, enable_array.count, index));
			}
			putBoard(board, put_value, current_color);
			//printBoard(board);
			if (calcReward(board, 999, 999, effort) >= 0.9) {
				win++;
			}
			effort++;

			current_color *= -1;
		}
	}
	printf("%d\n", win);
	return 0;
}