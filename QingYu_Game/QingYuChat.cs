using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using TMPro; // 新增TMP命名空间
/// <summary>
/// 清雨宗 · 雨司堂：墨清影对话客户端。
/// 适配 TextMeshPro(TMP) 输入框和文本
/// </summary>
public class QingYuChat : MonoBehaviour
{
    [Header("UI 引用")]
    [SerializeField] private TMP_InputField inputField;   // TMP输入框
    [SerializeField] private TMP_Text chatText;           // TMP对话文本
    [SerializeField] private Button sendButton;           // 发送按钮（普通Button不变）
    [SerializeField] private TMP_Text loadingText;        // TMP加载提示文本

    [Header("网络配置")]
    [SerializeField] private string serverUrl = "http://127.0.0.1:8000/chat";
    [SerializeField] private int timeoutSeconds = 90;

    private const string LoadingTip = "墨清影正在翻阅典籍，稍候……";
    private bool isRequesting;

    private void Awake()
    {
        if (sendButton != null)
        {
            sendButton.onClick.AddListener(SendQuestion);
        }
        if (loadingText != null)
        {
            loadingText.gameObject.SetActive(false);
        }
    }

    private void OnDestroy()
    {
        if (sendButton != null)
        {
            sendButton.onClick.RemoveListener(SendQuestion);
        }
    }

    public void SendQuestion()
    {
        if (isRequesting)
        {
            return;
        }
        if (inputField == null || chatText == null)
        {
            Debug.LogError("[QingYuChat] 请先在 Inspector 中绑定 InputField 与 chatText。");
            return;
        }

        string question = inputField.text.Trim();
        if (string.IsNullOrEmpty(question))
        {
            chatText.text = "弟子还未开口呢，请先写下想问的事。";
            inputField.ActivateInputField();
            return;
        }

        inputField.text = string.Empty;
        StartCoroutine(SendRequest(question));
    }

    private IEnumerator SendRequest(string question)
    {
        SetRequesting(true);
        chatText.text = LoadingTip;

        ChatRequest payload = new ChatRequest { question = question };
        string json = JsonUtility.ToJson(payload);
        byte[] body = Encoding.UTF8.GetBytes(json);

        using (UnityWebRequest request = new UnityWebRequest(serverUrl, UnityWebRequest.kHttpVerbPOST))
        {
            request.uploadHandler = new UploadHandlerRaw(body);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.timeout = timeoutSeconds;

            Debug.Log($"[QingYuChat] 发送提问：{question}");
            yield return request.SendWebRequest();
            HandleResponse(request);
        }
        SetRequesting(false);
    }

    private void HandleResponse(UnityWebRequest request)
    {
        if (request.result == UnityWebRequest.Result.ConnectionError)
        {
            bool isTimeout = !string.IsNullOrEmpty(request.error)
                && request.error.IndexOf("timeout", StringComparison.OrdinalIgnoreCase) >= 0;
            Debug.LogWarning($"[QingYuChat] 连接失败：{request.error}");
            chatText.text = isTimeout
                ? "墨清影久未回话，许是路上耽搁了，稍后再试。"
                : "墨清影未能应答，网络似乎不通，请稍后再试。";
            return;
        }

        if (request.result == UnityWebRequest.Result.ProtocolError)
        {
            Debug.LogWarning($"[QingYuChat] HTTP {request.responseCode}：{request.downloadHandler.text}");
            chatText.text = ExtractDetail(request.downloadHandler.text);
            return;
        }

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogWarning($"[QingYuChat] 请求异常：{request.error}");
            chatText.text = "传来的回信看不真切，请稍后再试。";
            return;
        }

        try
        {
            ChatResponse response = JsonUtility.FromJson<ChatResponse>(request.downloadHandler.text);
            chatText.text = response != null && !string.IsNullOrEmpty(response.answer)
                ? response.answer
                : "墨清影沉默不语，许是无从作答。";
        }
        catch (Exception exception)
        {
            Debug.LogError($"[QingYuChat] 解析回信失败：{exception.Message}，原文：{request.downloadHandler.text}");
            chatText.text = "回信的字迹有些模糊，请稍后再试。";
        }
    }

    private static string ExtractDetail(string responseBody)
    {
        try
        {
            ErrorResponse error = JsonUtility.FromJson<ErrorResponse>(responseBody);
            if (error != null && !string.IsNullOrEmpty(error.detail))
            {
                return error.detail;
            }
        }
        catch (Exception)
        {
        }
        return "雨司堂这边出了些岔子，请稍后再试。";
    }

    private void SetRequesting(bool requesting)
    {
        isRequesting = requesting;
        if (sendButton != null)
        {
            sendButton.interactable = !requesting;
        }
        if (loadingText != null)
        {
            loadingText.text = LoadingTip;
            loadingText.gameObject.SetActive(requesting);
        }
        if (!requesting && inputField != null)
        {
            inputField.ActivateInputField();
        }
    }

    [Serializable]
    private class ChatRequest
    {
        public string question;
    }
    [Serializable]
    private class ChatResponse
    {
        public string answer;
    }
    [Serializable]
    private class ErrorResponse
    {
        public string detail;
    }
}
