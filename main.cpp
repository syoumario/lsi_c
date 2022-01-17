#include<stdio.h>
#include <stdlib.h>
#include<math.h>
#include <time.h>
#include <iostream>
#include <tkusbfx3.h>
#include <windows.h>
#pragma comment(lib, "tkusbfx3.lib")
using namespace std;

#define BOARD_SIZE 8 //盤面の一辺の数
#define EPISODE 20000 //エピソード数
#define MEMORY_SIZE 800 //一度に保存するExperience Replyの数
#define BATCH_SIZE 512 //バッチサイズ
#define EPISODE_INTERVAL 80 //学習を行う頻度
#define MATCH_INTERVAL 500 //学習の合間に試合を行う頻度

#define INPUT_DIM 5 //入力次元（状態数）
#define MIDDLE_DIM 64 //隠れ層の次元
#define OUTPUT_DIM 64 //出力次元

typedef struct {
	int x;
	int y;
} coordinate;

//宣言時に初期化忘れない(0で初期化)
typedef struct {
	int enable[64]; //1だと置ける
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

typedef struct {
	float error;
	int epoch;
} history;
int history_index = 0;
#define SAVE_HISTORY_NAME "history.csv";
#define SAVE_MIDDLE_WEIGHT_NAME "middle_weight.csv"
#define SAVE_FINAL_WEIGHT_NAME "final_weight.csv"
#define SAVE_PROGRESS_RATE_NAME "progress_rate.csv"
#define SAVE_EPOCH_INTERVAL 100

//進行方向
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

const float board_score[64] = {
	0.3,-0.12,0,-0.01,-0.01,0,-0.12,0.3,
	-0.12,-0.15,-0.03,-0.03,-0.03,-0.03,-0.15,-0.12,
	0,-0.03,0,-0.01,-0.01,0,-0.03,0,
	-0.01,-0.03,-0.01,-0.01,-0.01,-0.01,-0.03,-0.01,
	-0.01,-0.03,-0.01,-0.01,-0.01,-0.01,-0.03,-0.01,
	0,-0.03,0,-0.01,-0.01,0,-0.03,0,
	-0.12,-0.15,-0.03,-0.03,-0.03,-0.03,-0.15,-0.12,
	0.3,-0.12,0,-0.01,-0.01,0,-0.12,0.3
};

//	q値(降順)
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

//ランダム行動:2、Q値から1
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
* state[0]：自分の角の石の数
* state[1]：相手の角の石の数
* state[2]：自分の石の数-相手の石の数
* state[3]：石の置かれていない場所の数
* state[4]：不利になる場所に置かれている石の数：自分-相手
*/
void createState(int* board, int* state) {
	for (int i = 0; i < INPUT_DIM;i++) state[i] = 0;

	for (int i = 0;i < BOARD_SIZE * BOARD_SIZE; i++) {
		if (i == 0 || i == 7 || i == 56 || i == 63) { //角
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
		if (black > white) reward += 10;
		else if (black <= white) reward -= 10;
	}
	else if(white == 0 || black == 0){
		if (white == 0) reward += 10;
		if (black == 0) reward -= 10;
	}

	//if (black_put_number == 0 || black_put_number == 7 || black_put_number == 56 || black_put_number == 63) reward += 0.2;
	////HACK:見づらい
	//else if (black_put_number == 1 || black_put_number == 8 || black_put_number == 9 || black_put_number == 6 || black_put_number == 14 || black_put_number == 15 || black_put_number == 48
	//	|| black_put_number == 49 || black_put_number == 57 || black_put_number == 54 || black_put_number == 55 || black_put_number == 62) reward -= 0.1;

	for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
		reward += board[i] * board_score[i];
	}

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

順伝搬計算

************************/

//OPTIMIZE:高速化可能
void calcForwardMiddleClass(int* input, float* output, float* weight, int input_dim, int output_dim) {
	for (int i = 0; i < output_dim; i++) output[i] = 0;

	for (int j = 0; j < input_dim * output_dim; j++) output[j / input_dim] += input[j % input_dim] * weight[j];

	for (int k = 0; k < output_dim; k++) output[k] = tanhf(output[k]);
}

//OPTIMIZE:高速化可能
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

順伝搬計算終わり

************************/

/************************

誤差逆伝搬

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

void calcErrorBackPropagation(int* input, float* d3, float* middle_output, float* final_weight,  float* final_delta, float* middle_delta, int input_dim, int final_dim, int middle_dim) {
	for (int i = 0; i < final_dim * middle_dim; i++) {
		final_delta[i] = d3[i % final_dim] * middle_output[i / middle_dim];
	}

	float tmp_middle[MIDDLE_DIM * OUTPUT_DIM] = { 0 };
	for (int j = 0; j < middle_dim * final_dim; j++) {
		tmp_middle[j / middle_dim] += d3[j % final_dim] * final_weight[j] / powf(coshf(atanhf(middle_output[j / middle_dim])), 2);
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

void doTrainQNetwork(history* history, experience_reply* reply, float* middle_weight, float* final_weight, int input_dim, int middle_dim, int output_dim) {
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
			if (i == batch[batch_index].action) {
				diff_q_value[i] = - q_value[i] + batch[batch_index].reward + 0.99 * q_value_with_index[0].value;
				if (batch_index % SAVE_EPOCH_INTERVAL == 0)history[history_index].error = diff_q_value[i];
			}
			else
				diff_q_value[i] = 0;
		}
		if (batch_index % SAVE_EPOCH_INTERVAL == 0)history[history_index].epoch = history_index;
		if (batch_index % SAVE_EPOCH_INTERVAL == 0)history_index++;
		calcErrorBackPropagation(batch[batch_index].state, diff_q_value, middle_output, final_weight, final_delta, middle_delta, input_dim, output_dim, middle_dim);
		updateWeight(middle_weight, final_weight, middle_delta, final_delta, 0.25, input_dim, middle_dim, output_dim);
	}
}

/************************

誤差逆伝搬終わり

************************/

/************************

Artix7で学習

************************/

bool UsbOpen()	//デバイスUSBのオープン関数
{
	char DeviceName[100];
	unsigned short pid, vid;

	for (int timeout = 0; timeout < 3; timeout++)
	{
		bool status = TKUSBFX3Open(0, &vid, &pid, DeviceName, sizeof(DeviceName));

		if (status)
		{
			//printf("USB open success. VID=%04x PID=%04x DeviceName=\"%s\"\n", vid, pid, DeviceName);
		}
		else
		{
			printf("USB open failed.\n");
			return false;
		}

		if ((vid == 0x2129) && (pid == 0x0520))
		{
			printf("NP1053(特電FX3ボード)を発見しました\n");
			return true;
		}
		else if ((vid == 0x2129) && (pid == 0x0640))
		{
			//printf("NP1064(特電Artix-7ボード)を発見しました\n");
			return true;
		}
		else
		{
			char ErrorReason[100];
			printf("異なるVID,PIDのデバイスが見つかったので、ファームウェアを書き込みます。\n");
			if (TKUSBFX3WriteToRAM("SlaveFifoNP1064.img", ErrorReason, 100))
			{
				printf("ファームウェアを書き込みました\n");
				Sleep(1000);
			}
			else
			{
				printf("ファームウェアの転送に失敗しました 理由:%s\n", ErrorReason);
				return false;
			}
		}
	}
	return false;
}

void trans_2bit16array_from_double(double val_t, int out[16]) {
	double val;
	int minusflag = 0;
	if (val_t < 0) {
		val = val_t * (-1);
		minusflag = 1;
	}
	else {
		val = val_t;
	}
	int x_i = int(val);
	double x_f = val - x_i;

	out[0] = 0;

	//整数部を5bitに変換
	for (int i = 0; i < 5; i++) {
		out[5 - i] = x_i % 2;
		x_i /= 2;
	}

	//小数部を10bitに変換
	for (int i = 0; i < 10; i++) {
		out[6 + i] = int(x_f * 2);
		x_f = x_f * 2 - out[6 + i];
	}

	//補数をとる
	if (minusflag == 1) {
		//ビットの反転
		for (int i = 0; i < 16; i++) {
			if (out[i] == 1) out[i] = 0;
			else out[i] = 1;
		}
		//最下位ビットに+1
		if (out[15] == 0) out[15] = 1;
		else {
			out[15] = 0;
			for (int i = 0; i < 15; i++) {
				if (out[14 - i] == 0) {
					out[14 - i] = 1;
					break;
				}
				else out[14 - i] = 0;
			}
		}
	}

}

void trans_2bit8array_from_unsignedchar(unsigned char val_t, int out[8]) {
	unsigned char val = val_t;

	//正数を8bitに変換
	for (int i = 0; i < 8; i++) {
		out[7 - i] = val % 2;
		val /= 2;
	}

}

double trans_double_from_2bit16array(int in[16]) {
	double val = 0;
	int x[16];

	if (in[0] == 0) {
		for (int i = 0; i < 15; i++) {
			val += in[15 - i] * pow(2, (i - 10));
		}
		return val;
	}
	else {
		//ビットの反転
		for (int i = 0; i < 16; i++) {
			if (in[i] == 1) x[i] = 0;
			else x[i] = 1;
		}
		//最下位ビットに+1
		if (x[15] == 0) x[15] = 1;
		else {
			x[15] = 0;
			for (int i = 0; i < 15; i++) {
				if (x[14 - i] == 0) {
					x[14 - i] = 1;
					break;
				}
				else x[14 - i] = 0;
			}
		}

		for (int i = 0; i < 15; i++) {
			val += x[15 - i] * pow(2, (i - 10));
		}

		val *= -1;

		return val;

	}
}

unsigned char trans_unsignedchar_from_8bitarray(int in[8]) {
	unsigned char out;
	int x = 0;
	for (int i = 0; i < 8; i++) {
		x += in[7 - i] * pow(2, i);
	}
	out = unsigned char(x);
	return out;
}

void trans_state(int state[5], unsigned char out_state[10]) {

	int val_16bit[16];
	double val;

	for (int i = 0; i < 5; i++) {
		val = double(state[i]);

		trans_2bit16array_from_double(val, val_16bit);

		int val_8bit_up[8];
		int val_8bit_under[8];

		for (int i = 0; i < 8; i++) {
			val_8bit_up[i] = val_16bit[i];
		}
		for (int i = 0; i < 8; i++) {
			val_8bit_under[i] = val_16bit[i + 8];
		}

		unsigned char x_up = trans_unsignedchar_from_8bitarray(val_8bit_up);
		unsigned char x_under = trans_unsignedchar_from_8bitarray(val_8bit_under);

		out_state[2 * i] = x_under;
		out_state[2 * i + 1] = x_up;
	}

}

void trans_reward(double reward, unsigned char out_state[2]) {
	int val_16bit[16];
	double val;

	val = double(reward);

	trans_2bit16array_from_double(val, val_16bit);

	int val_8bit_up[8];
	int val_8bit_under[8];

	for (int i = 0; i < 8; i++) {
		val_8bit_up[i] = val_16bit[i];
	}
	for (int i = 0; i < 8; i++) {
		val_8bit_under[i] = val_16bit[i + 8];
	}

	unsigned char x_up = trans_unsignedchar_from_8bitarray(val_8bit_up);
	unsigned char x_under = trans_unsignedchar_from_8bitarray(val_8bit_under);

	out_state[0] = x_under;
	out_state[1] = x_up;

}

void trans_double_fromArtix222(unsigned char in[160], double out[80]) {
	int val_16bit[16];
	int val_8bit_up[8];
	int val_8bit_under[8];
	for (int i = 0; i < 80; i++) {
		trans_2bit8array_from_unsignedchar(in[i * 2], val_8bit_under);
		trans_2bit8array_from_unsignedchar(in[i * 2 + 1], val_8bit_up);

		for (int i = 0; i < 8; i++) {
			val_16bit[i] = val_8bit_up[i];
			val_16bit[i + 8] = val_8bit_under[i];
		}

		out[i] = trans_double_from_2bit16array(val_16bit);

	}
}

void trans_action(double reward, unsigned char out_state[2]) {
	int val_16bit[16];
	double val;

	val = double(reward);

	int minusflag = 0;

	int x_i = int(val);

	double x_f = val - x_i;

	val_16bit[0] = 0;

	//整数部を5bitに変換
	for (int i = 0; i < 6; i++) {
		val_16bit[6 - i] = x_i % 2;
		x_i /= 2;
	}

	//小数部を10bitに変換
	for (int i = 0; i < 9; i++) {
		val_16bit[7 + i] = int(x_f * 2);
		x_f = x_f * 2 - val_16bit[7 + i];
	}

	int val_8bit_up[8];
	int val_8bit_under[8];

	for (int i = 0; i < 8; i++) {
		val_8bit_up[i] = val_16bit[i];
	}
	for (int i = 0; i < 8; i++) {
		val_8bit_under[i] = val_16bit[i + 8];
	}

	unsigned char x_up = trans_unsignedchar_from_8bitarray(val_8bit_up);
	unsigned char x_under = trans_unsignedchar_from_8bitarray(val_8bit_under);

	out_state[0] = x_under;
	out_state[1] = x_up;

}

void new_trans_middle_weight(float* midle_weight, unsigned char* out, int input_dim, int middle_dim) {

	int val_16bit[16];
	double val;
	for (int i = 0; i < input_dim * middle_dim; i++) {
		val = double(midle_weight[i]);

		trans_2bit16array_from_double(val, val_16bit);

		int val_8bit_up[8];
		int val_8bit_under[8];

		for (int i = 0; i < 8; i++) {
			val_8bit_up[i] = val_16bit[i];
		}
		for (int i = 0; i < 8; i++) {
			val_8bit_under[i] = val_16bit[i + 8];
		}

		unsigned char x_up = trans_unsignedchar_from_8bitarray(val_8bit_up);
		unsigned char x_under = trans_unsignedchar_from_8bitarray(val_8bit_under);

		out[i * 2] = x_under;
		out[i * 2 + 1] = x_up;
	}

}

void new_trans_final_weight(float* final_weight, unsigned char* out, int middle_dim, int final_dim) {

	int val_16bit[16];
	double val;
	for (int i = 0; i < middle_dim * final_dim; i++) {
		val = double(final_weight[i]);

		trans_2bit16array_from_double(val, val_16bit);

		int val_8bit_up[8];
		int val_8bit_under[8];

		for (int i = 0; i < 8; i++) {
			val_8bit_up[i] = val_16bit[i];
		}
		for (int i = 0; i < 8; i++) {
			val_8bit_under[i] = val_16bit[i + 8];
		}

		unsigned char x_up = trans_unsignedchar_from_8bitarray(val_8bit_up);
		unsigned char x_under = trans_unsignedchar_from_8bitarray(val_8bit_under);

		out[i * 2] = x_under;
		out[i * 2 + 1] = x_up;
	}

}

void sendData_doTrain(experience_reply* exp, float* midle_weight, float* final_weight, int input_dim, int middle_dim, int output_dim) {

	const int DATASIZE = 320;
	unsigned char X[DATASIZE] = { 0 };
	const int inp = 5;
	const int mid = 64;
	const int out = 64;

	unsigned char state[10];
	unsigned char next_state[10];
	//行動のみ、1-6-9最大値が35であるため
	unsigned char act[2];
	unsigned char reward[2];
	unsigned char middle_weight_char[inp * mid * 2];
	unsigned char final_weight_char[mid * out * 2];

	new_trans_middle_weight(midle_weight, middle_weight_char, input_dim, middle_dim);
	new_trans_final_weight(final_weight, final_weight_char, middle_dim, output_dim);

	//開始.デバイスUSBのオープン
	//printf("プログラム起動.デバイス接続を確認します\n\n");
	if (UsbOpen()) {
		//printf("デバイス発見完了.%d個のデバイスが発見されました\n\n", TKUSBFX3DeviceCount());
		if (TKUSBFX3DeviceCount() != 1) {
			printf("このプログラムは2個以上のデバイスに対応していません.よって終了します\n");
		}
	}
	else {
		printf("デバイスが発見できません.FPGAを接続してください\n");
	}
	int USB_R, USB_W;

	//経験を送信：X[0] = 1.X[1] = .X[2] and X[3] = 未定.
	for (int c = 0; c < 512; c++) {
		trans_state(exp[c].state, state);
		trans_state(exp[c].next_state, next_state);
		trans_action(exp[c].action, act);
		trans_reward(exp[c].reward, reward);


		X[0] = 1;
		X[1] = 0;

		X[2] = 0;
		X[3] = 0;

		/* set data */
		for (int i = 0; i < 10; i++) {
			X[4 + i] = state[i];
		}
		for (int i = 0; i < 10; i++) {
			X[4 + 10 + i] = next_state[i];
		}
		for (int i = 0; i < 2; i++) {
			X[4 + 20 + i] = act[i];
		}
		for (int i = 0; i < 2; i++) {
			X[4 + 22 + i] = reward[i];
		}

		if (c < 64) {
			for (int i = 0; i < 10; i++) {
				X[4 + 22 + 2 + i] = middle_weight_char[c * 10 + i];
			}
			for (int i = 0; i < 128; i++) {
				X[4 + 22 + 2 + 10 + i] = final_weight_char[c * 128 + i];
			}
		}

		/*特電IPコアにデータ送信*/
		unsigned short flag = 0;//送信のオプション
		unsigned long addr = 0x10000;

		USB_W = USBWriteData(addr, X, DATASIZE, flag);
		if (USB_W == 0)
			printf("FPGAへのデータ送信に失敗しました\n\n");

		//Sleep(1000);
	}

	X[0] = 2;
	X[1] = 0;

	/*特電IPコアにデータ送信*/
	unsigned short flag = 0;//送信のオプション
	unsigned long addr = 0x10000;

	USB_W = USBWriteData(addr, X, DATASIZE, flag);
	if (USB_W == 0)
		printf("FPGAへのデータ送信に失敗しました\n\n");
	//else
		//printf("学習スタート\n\n", USB_W);

}

void doTrainQNetwork_byartix(history* history, experience_reply* reply, float* middle_weight, float* final_weight, int input_dim, int middle_dim, int output_dim) {

	experience_reply exp[BATCH_SIZE];
	float target_middle_weight[MIDDLE_DIM * INPUT_DIM], target_final_weight[MIDDLE_DIM * OUTPUT_DIM];
	float target_q_value[OUTPUT_DIM], q_value[OUTPUT_DIM], diff_q_value[OUTPUT_DIM];
	float middle_output[MIDDLE_DIM];
	float middle_delta[MIDDLE_DIM * INPUT_DIM], final_delta[MIDDLE_DIM * OUTPUT_DIM];
	pullOutExperienceMemory(reply, exp, BATCH_SIZE, MEMORY_SIZE);

	//データの送信と学習のスタート
	sendData_doTrain(exp, middle_weight, final_weight, INPUT_DIM, MIDDLE_DIM, OUTPUT_DIM);

	Sleep(100);

	/*特電IPコアからデータ受信*/
	static unsigned char X_re[160] = {};//受信データ格納配列
	//受信結果変換後
	double TT[80] = { 0 };

	int USB_R, USB_W;
	int success = 1;
	while (TT[0] != 1) {
		for (int c = 0; c < 64; c++) {
			USB_R = USBReadData(0, X_re, 160, 1);
			if (USB_R == 0) {
				printf("Data reception failed so not update-Weight\n\n");
				success = 0;
			}
			trans_double_fromArtix222(X_re, TT);
			if (TT[0] = 1) {
				for (int i = 1; i < 70; i++) {
					//cout << TT[i] << ",";
				}
				//cout << "\n";
				//cout << "\n";
				for (int i = 0; i < 5; i++) {
					middle_delta[c * INPUT_DIM + i] = TT[1 + i];
					//cout << middle_weight[c * INPUT_DIM + i];
				}
				for (int i = 0; i < 64; i++) {
					final_delta[c * OUTPUT_DIM + i] = TT[6 + i];
				}
			}
			//else
				//cout << "・";
		}
	}

	if (success == 1) {
		for (int i = 0; i < INPUT_DIM * MIDDLE_DIM; i++) {
			middle_weight[i] = middle_weight[i] - (middle_weight[i] - middle_delta[i]);
		}
		for (int i = 0; i < MIDDLE_DIM * OUTPUT_DIM; i++) {
			final_weight[i] = final_weight[i] - (final_weight[i] - final_delta[i]);
		}
	}
	TKUSBFX3Close();

}


/************************

Artix7で学習終わり

************************/

/************************

学習パラメータの保存

************************/

void saveHistory(history* history) {
	FILE* fp;
	const char* fname = SAVE_HISTORY_NAME;
	errno_t error;

	int remove(*fname);

	error = fopen_s(&fp, fname, "w");
	if (fp == NULL) {
		printf("file open error ¥n");
	}
	else {

		for (int i = 0; i < history_index; i++) {
			std::cout << history[i].epoch << "," << history[i].error << "\n";
			fprintf(fp, "%d,%f\n", history[i].epoch, abs(history[i].error));
		}

		fclose(fp);

		printf("%sファイル書き込みが終わりました\n", fname);

	}

}

void saveWeight(float* weight, int length, const char* file_name) {
	FILE* fp;
	fopen_s(&fp, file_name, "w");
	
	if (fp == NULL) puts("ファイル開けません");
	else {
		for (int i = 0; i < length; i++) {
			fprintf(fp, "%lf,\n", weight[i]);
		}
	}
	fclose(fp);
}


/************************

学習パラメータの保存終わり

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
	puts(" １２３４５６７８");
	for (int i = 0; i < BOARD_SIZE; i++) {
		printf("%d", i);
		for (int j = 0; j < BOARD_SIZE; j++) {
			if (board[i * BOARD_SIZE + j] == -1) printf("●");
			else if (board[i * BOARD_SIZE + j] == 1) printf("○");
			else printf("ー");
		}
		puts("");
	}
}

void printEnablePut(enable_put enable_array) {
	for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) if(enable_array.enable[i] == 1) printf("%d、", i);
	puts("");
}

void printQValue(array_with_index* index, int count) {
	for (int i = 0;i < count; i++) printf("%d：%lf\n", index[i].index, index[i].value);
}

float playOthelloRate(int* board, int play_count, float* middle_weight, float* combined_weight) {
	int win = 0;
	array_with_index q_value_with_index[OUTPUT_DIM];
	float q_value[OUTPUT_DIM] = { 0 };

	for (int index = 0;index < play_count; index++) {
		resetBoard(board);
		int effort = 1;
		int current_color = 1;
		int pass = 0;
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

			createState(board, state); //ここをいじる

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
			effort++;

			current_color *= -1;
		}
		int tmp_stone_count = 0;
		for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
			tmp_stone_count += board[i];
		}
		if (tmp_stone_count > 0) win++;
	}

	return (float)win / play_count;
}

void saveProgressRate(array_with_index *input, int length, const char* file_name) {
	FILE* fp;
	fopen_s(&fp, file_name, "w");

	if (fp == NULL) puts("ファイル開けません");
	else {
		for (int i = 0; i < length; i++) {
			fprintf(fp, "%d,%lf,\n", input[i].index, input[i].value);
		}
	}
	fclose(fp);
}

int main() {
	int board[BOARD_SIZE * BOARD_SIZE] = { 0 }; //空きマス:0、白:-1、黒：1
	experience_reply memory[MEMORY_SIZE];
	float middle_weight[INPUT_DIM * MIDDLE_DIM], combined_weight[MIDDLE_DIM * OUTPUT_DIM];
	history history[BATCH_SIZE * EPISODE / EPISODE_INTERVAL / SAVE_EPOCH_INTERVAL] = { 0 };
	setUniformDistributionToArray(middle_weight, INPUT_DIM * MIDDLE_DIM, INPUT_DIM);
	setUniformDistributionToArray(combined_weight, MIDDLE_DIM * OUTPUT_DIM, MIDDLE_DIM);

	int effort = 1, current_color=1, memory_index = 0, pass=0;
	resetBoard(board);
	array_with_index q_value_with_index[OUTPUT_DIM];
	array_with_index progress_rate[EPISODE / MATCH_INTERVAL];
	int progress_index = 0;
	float q_value[OUTPUT_DIM] = { 0 }, max_rate = 0.0;
	setIndex(q_value_with_index, q_value, OUTPUT_DIM);

	for (int episode = 0; episode < EPISODE; episode++) {
		if (episode % (EPISODE / 10) == 0)std::cout << "・";
		if (episode % EPISODE_INTERVAL == 0 && episode != 0) {
			//doTrainQNetwork(history, memory, middle_weight, combined_weight, INPUT_DIM, MIDDLE_DIM, OUTPUT_DIM);
			doTrainQNetwork_byartix(history, memory, middle_weight, combined_weight, INPUT_DIM, MIDDLE_DIM, OUTPUT_DIM);
			resetEpisode(memory);
		}
		int prev_color = 0, now_color = 0, black_put_value = 999, white_put_value = 999;
		if (episode % MATCH_INTERVAL == 0) {
			progress_rate[progress_index].index = episode;
			progress_rate[progress_index].value = playOthelloRate(board, 500, middle_weight, combined_weight);
			if (progress_rate[progress_index].value > max_rate) {
				max_rate = progress_rate[progress_index].value;
				saveWeight(middle_weight, INPUT_DIM * MIDDLE_DIM, SAVE_MIDDLE_WEIGHT_NAME);
				saveWeight(combined_weight, MIDDLE_DIM * OUTPUT_DIM, SAVE_FINAL_WEIGHT_NAME);
				printf("重み保存 %d: %lf\n", episode, progress_rate[progress_index].value);
			}
			progress_index++;
		}
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

			createState(board, state); //ここをいじる

			if (current_color == 1) {
				const int action_term = selectEpisilonOrGreedy(0.9, 0.1, 35, episode);
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

	saveHistory(history);
	/*saveWeight(middle_weight, INPUT_DIM * MIDDLE_DIM, SAVE_MIDDLE_WEIGHT_NAME);
	saveWeight(combined_weight, MIDDLE_DIM * OUTPUT_DIM, SAVE_FINAL_WEIGHT_NAME);*/
	saveProgressRate(progress_rate, EPISODE / MATCH_INTERVAL, SAVE_PROGRESS_RATE_NAME);

	return 0;
}