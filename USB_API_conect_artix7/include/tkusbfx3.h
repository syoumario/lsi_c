/**************************************************************************//**
 * @file    tkusnfx3.h
 * @brief   特電USB FX3ライブラリ
 * @details EZ-USB FX3のCyAPIをラッパしてBCCで使えるようにしています
 * @todo	
 ******************************************************************************/

#include <Windows.h>

#ifndef TKUSBFX3_API
	#define TKUSBFX3_API __declspec(dllimport)
#endif

// このクラスは txusbfx3.dll からエクスポートされました。
class Ctxusbfx3 {
public:
	Ctxusbfx3(void);
	// TODO: メソッドをここに追加してください。
};

#ifdef __cplusplus
	extern "C" {
#endif

typedef enum 
{
	TKUSBFX3_SUCCESS                = 0,
	TKUSBFX3_NOT_OPEN               = -1,
	TKUSBFX3_CY_DEFAULT_FIRMWARE    = -2,
	TKUSBFX3_ENDPOINT_NOTFOUND      = -3,
	TKUSBFX3_LOCK_TIMEOUT           = -4,
	TKUSBFX3_XFER_ERROR             = -5
} TKUSBFX3_ERR;

/*********************************************************************//**
 * @brief		ライブラリをオープンする
 * @details		ライブラリをオープンする
 * @param		num 何のパラメータか忘れた。0を指定すればいい
 *              vid 見つかったデバイスのベンダIDが格納される
 *              pid 見つかったデバイスのプロダクトIDが格納される
 *              DeviceName 見つかったデバイスの名前が格納される
 *              MaxDevnameLength 名前が格納されるバッファの長さ
 * @return		見つかったらtrueを返す。見つからなかったらfalseを返す。
 **********************************************************************/
TKUSBFX3_API BOOL WINAPI TKUSBFX3Open(int num,unsigned short *vid,unsigned short *pid,char *DeviceName,int MaxDevnameLength);

/*********************************************************************//**
 * @brief		ライブラリをクローズする
 * @details		ライブラリをクローズする
 * @param		なし
 * @return		なし
 **********************************************************************/
TKUSBFX3_API void  WINAPI TKUSBFX3Close(void);

/*********************************************************************//**
 * @brief		見つかったFX3デバイスの数を返す
 * @details		見つかったFX3デバイスの数を返す
 * @param		なし
 * @return		見つかったデバイスの数
 **********************************************************************/
TKUSBFX3_API int   WINAPI TKUSBFX3DeviceCount(void);

/*********************************************************************//**
 * @brief		FX3の内蔵RAMにイメージファイルを書き込む
 * @details		FX3の内蔵RAMにイメージファイルを書き込む
 * @param		ImageFileName 書き込むイメージファイル(*.img)
                ErrorReason エラーの原因が文字列で入るバッファ
                ErrorReasonLength 文字列バッファの長さ
 * @return		成功したらtrueを返す
 **********************************************************************/
TKUSBFX3_API BOOL  WINAPI TKUSBFX3WriteToRAM(const char *ImageFileName,char *ErrorReason, int ErrorReasonLength);

/*********************************************************************//**
 * @brief		FX3のSPI ROMにイメージファイルを書き込む
 * @details		FX3のSPI ROMにイメージファイルを書き込む
 * @param		ImageFileName 書き込むイメージファイル(*.img)
                ErrorReason エラーの原因が文字列で入るバッファ
                ErrorReasonLength 文字列バッファの長さ
 * @return		成功したらtrueを返す
 **********************************************************************/
TKUSBFX3_API BOOL  WINAPI TKUSBFX3WriteToSPIROM(const char *ImageFileName,char *ErrorReason, int ErrorReasonLength);

/*********************************************************************//**
 * @brief		低レベルなバルクOUT
 * @details		FX3のAPIを呼び出し、任意のEndPointに任意の長さのデータを転送する
 * @param		ep エンドポイント番号
                data 送信したいデータ
                length 送信したいデータの長さ(バイト単位)
 * @return		成功したら、送信した長さを返す。
                USBデバイスがなければ-1を返す。
                EZ-USB Loaderのままであれば-2を返す。
                データ転送で失敗したら-3を返す。
 * @note		1024バイト単位で送信すると、バッファに溜まってうまく送信されないことがある
                128バイトとか、短い単位で送ると、バッファにたまらずにすぐに送信されるようだ
 **********************************************************************/
TKUSBFX3_API int   WINAPI TKUSBFX3BulkOut(int ep,unsigned char *data,int length);

/*********************************************************************//**
 * @brief		低レベルなバルクIn
 * @details		FX3のAPIを呼び出し、任意のEndPointから任意の長さのデータを受信する
 * @param		ep エンドポイント番号
                data 受信したデータが格納されるバッファ
                length 受信したいデータの長さ(バイト単位)
 * @return		成功したら、送信した長さを返す。
                USBデバイスがなければ-1を返す。
                EZ-USB Loaderのままであれば-2を返す。
                データ転送で失敗したら-3を返す。
 **********************************************************************/
TKUSBFX3_API int   WINAPI TKUSBFX3BulkIn(int ep,unsigned char *data,int length);

/*********************************************************************//**
 * @brief		高レベルな送信(BulkOut)
 * @details		FPGAの中のIPコアを呼び出し、データを送信する。
 * @param		addr コアから出力されるaddr信号の値
                data 送信したいデータが格納されるバッファ
                len 送信したいデータの長さ
                flag コアから出力されるflag信号の値
 * @return		成功したら、送信した長さを返す。
                失敗したら、0を返す。
 **********************************************************************/
TKUSBFX3_API int   WINAPI USBWriteData(unsigned long addr,unsigned char *data,int len,unsigned short flag);

/*********************************************************************//**
 * @brief		高レベルな受信(BulkIn)
 * @details		FPGAの中のIPコアを呼び出し、データを受信する。
 * @param		addr コアから出力されるaddr信号の値
                data 受信したいデータが格納されるバッファ
                len 受信したいデータの長さ(4の倍数に切り上げられる)
                flag コアから出力されるflag信号の値
 * @return		成功したら、送信した長さを返す。
                失敗したら、0を返す。
 **********************************************************************/
TKUSBFX3_API int   WINAPI USBReadData(unsigned long addr,unsigned char *data,int len,unsigned short flag);

/*********************************************************************//**
 * @brief		高レベルな受信(BulkIn)
 * @details		FPGAの中のIPコアを呼び出し、データを受信する。
 * @param		addr コアから出力されるaddr信号の値
                data 受信したいデータが格納されるバッファ
                len 受信したいデータの長さ(4の倍数に切り上げられる)
                flag コアから出力されるflag信号の値
 * @return		成功したら、送信した長さを返す。
                失敗したら、0を返す。
 **********************************************************************/
TKUSBFX3_API int   WINAPI USBReadData(unsigned long addr,unsigned char *data,int len,unsigned short flag);

/*********************************************************************//**
 * @brief		リセット
 * @details		USBデバイスのリセットと、USBエンドポイントのリセット
 * @param		なし
 * @return		なし
 **********************************************************************/
TKUSBFX3_API void  WINAPI USBReset(void);

/*********************************************************************//**
 * @brief		タイムアウトの設定
 * @details		USBから応答がない場合のタイムアウト時間を設定する。デフォルトは10秒。
 * @param		milliseconds ミリ秒単位でバルク入出力のタイムアウトを設定する。0xFFFFFFFFを設定すると無限に待つ。
 * @return		なし
 **********************************************************************/
TKUSBFX3_API void  WINAPI TKUSBSetTimeout(unsigned long milliseconds);

/*********************************************************************//**
 * @brief		最大転送サイズの設定
 * @details		1回の転送で何バイトを転送するか。デフォルトは1048576バイト。
 * @param		bytes 転送サイズ。2MB以上にすると特定のマシンでハングアップする可能性がある。
 * @return		なし
 **********************************************************************/
TKUSBFX3_API void  WINAPI TKUSBSetTransferSize(int bytes);

#ifdef __cplusplus
	}
#endif
