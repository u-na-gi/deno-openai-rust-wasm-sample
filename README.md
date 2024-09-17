# aimaid


アシスタント作成

```shell
curl "https://api.openai.com/v1/assistants"   -H "Content-Type: application/json"   -H "Authorization: Bearer $OPENAI_API_KEY"   -H "OpenAI-Beta: assistants=v2"   -d '{
    "instructions": "You are an excellent software engineer with expertise in frontend development, backend development, algorithms, testing, documentation, and architecture. Please use your skills to approach the following task.",
    "name": "Softwhere Engineer",
    "tools": [{"type": "code_interpreter"}],
    "model": "gpt-4o-mini"
}'
```

thread作成
```shell
curl https://api.openai.com/v1/threads   -H "Content-Type: application/json"   -H "Auhorization: Bearer $OPENAI_API_KEY"   -H "OpenAI-Beta: assistants=v2"   -d ''
```

threadにメッセージを入れる
```shell
curl https://api.openai.com/v1/threads/<上で作成したthread id>/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "OpenAI-Beta: assistants=v2" \
  -d '{
      "role": "user",
      "content": "I need to solve the equation `3x + 11 = 14`. Can you help me?"
    }'

# 例
curl https://api.openai.com/v1/threads/thread_pSq4JbJ9m1h41ksA9wt3ZDwc/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "OpenAI-Beta: assistants=v2" \
  -d '{
      "role": "user",
      "content": "I need to solve the equation `3x + 11 = 14`. Can you help me?"
    }'
```

run

```shell
curl https://api.openai.com/v1/threads/<上で作成したthread id>/runs \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -H "OpenAI-Beta: assistants=v2" \
  -d '{
    "assistant_id": "<作成したssistant>",
    "instructions": "Please address the user as Jane Doe. The user has a premium account."
  }'

# 例
curl https://api.openai.com/v1/threads/thread_pSq4JbJ9m1h41ksA9wt3ZDwc/runs \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -H "OpenAI-Beta: assistants=v2" \
  -d '{
    "assistant_id": "asst_F01GLhl3mov7DrQcTtv3BTHz",
    "instructions": "Please address the user as Jane Doe. The user has a premium account. Provide the response in JSON format."
  }'
```

jsonで返してもらうには、instructionsに`Provide the response in JSON format.`と書く必要があるらしい。

curl の場合ポーリング
```shell
curl https://api.openai.com/v1/threads/<上で作成したthread id>/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "OpenAI-Beta: assistants=v2"

# 例
curl https://api.openai.com/v1/threads/thread_pSq4JbJ9m1h41ksA9wt3ZDwc/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "OpenAI-Beta: assistants=v2"
```

nodejsで書くと非同期で待てるっぽいからそっちのがいいかもね

よってdenoで書く。