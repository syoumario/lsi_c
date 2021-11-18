#include<stdio.h>
#include <stdlib.h>
#include<math.h>
#include <time.h>

#define BOARD_SIZE 8
#define EPISODE 1600

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
	int state[64];
	int action;
	int next_state[64];
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

void setUniformDistributionToArray(float* output, int count) {
	srand((unsigned int)time(NULL));

	for (int i = 0; i < count; i++) output[i] = (rand() + 0.5) / (RAND_MAX + 1);
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

void putBoard(int* board, int number, int current_color) {
	board[number] = current_color;
	float reward = 0;
	for (int dir = 0; dir < 8; dir++) {
		int count = 0;
		for (int dir_value = 1; dir_value <= 6; dir_value++) {
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

float calcReward(int* board, int put_number, int effort) {
	float reward = 0.0;
	int black = 0, white = 0;
	if (effort == 60) {
		for (int index = 0; index < BOARD_SIZE * BOARD_SIZE, index++) {
			if (board[index] == 1) black++;
			else white++;
		}
		if (black > white) reward++;
		else if (black < white) reward--;
	}

	if (put_number == 0 || put_number == 7 || put_number == 56 || put_number == 63) reward += 0.2;
	//HACK:Œ©‚Ã‚ç‚¢
	else if (put_number == 1 || put_number == 8 || put_number == 9 || put_number == 6 || put_number == 14 || put_number == 15 || put_number == 48
		|| put_number == 49 || put_number == 57 || put_number == 54 || put_number == 55 || put_number == 62) reward -= 0.1;

	return reward;
}

int choicePutValue(enable_put enable_array, array_with_index* q_value) {
	for (int q_value_index = 0; q_value_index < BOARD_SIZE * BOARD_SIZE; q_value_index++) {
		if (enable_array.enable[q_value[q_value_index].index] == 1) return q_value[q_value_index].index;
	}

	return -1;
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
}

void calcForwardpropagation(int* input, float* output, float* weight_middle, float* weight_full, int input_dim, int middle_dim, int output_dim) {
	float *middle_output;
	middle_output = (float*)malloc(middle_dim * sizeof(float));
	calcForwardMiddleClass(input, middle_output, weight_middle, input_dim, middle_dim);
	calcForwardFullcombined(middle_output, output, weight_full, middle_dim, output_dim);
}

/************************

‡“`”ÀŒvŽZI‚í‚è

************************/

void resetBoard(int* board) {
	for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
		if (i == 27 || i == 36) board[i] = -1;
		else if (i == 28 || i == 35) board[i] = 1;
		else board[i] = 0;
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

int main() {
	int board[BOARD_SIZE * BOARD_SIZE] = { 0 }; //‹ó‚«ƒ}ƒX:0A”’:-1A•F1
	float middle_weight[BOARD_SIZE * BOARD_SIZE * BOARD_SIZE  * BOARD_SIZE], combined_weight[BOARD_SIZE * BOARD_SIZE];
	setUniformDistributionToArray(middle_weight, BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE);
	setUniformDistributionToArray(combined_weight, BOARD_SIZE * BOARD_SIZE);

	int effort = 1, current_color=1;
	resetBoard(board);
	array_with_index q_value_with_index[BOARD_SIZE * BOARD_SIZE];
	float q_value[BOARD_SIZE * BOARD_SIZE] = { 0 };
	setIndex(q_value_with_index, q_value, BOARD_SIZE * BOARD_SIZE);

	for (int episode = 0; episode < EPISODE; episode++) {
		while (effort <= 60) {
			const enable_put enable_array = checkPutCapability(board, current_color);

			if (enable_array.count == 0) {
				current_color *= -1;
				continue;
			}

			if()
			qsort(q_value_with_index, BOARD_SIZE * BOARD_SIZE, sizeof(array_with_index), cmpDescValue);
			const int put_value = choicePutValue(enable_array, q_value_with_index);
			putBoard(board, put_value, current_color);
			effort++;
			const float reward = calcReward(board, put_value, effort);
			current_color *= -1;
			printBoard(board);
		}
	}

	return 0;
}