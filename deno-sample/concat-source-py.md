
- /app/deno-sample/doll/engine/core/example/check_speaker_size_and_delete_from_memmory.py
```python
import numpy as np

from doll.speaker.value_objects.source import SpeakerSource


def print_size(obj):
    from pympler import asizeof

    size_in_bytes = asizeof.asizeof(obj)
    size_in_gb = size_in_bytes / (1024**3)
    return f"Size in GB: {size_in_gb:.10f} GB"


sr = 16000

ss = SpeakerSource(
    np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), sr
).ref()

state = ss.serialize()

a = ss.deserialize(state)

print(a)

del ss

```

- /app/deno-sample/doll/engine/core/tests/value_objects/test_serdes.py
```python
from doll_core.serdes import serdes
from dataclasses import dataclass


def test_UserFrozen_serialize_deserialize():
    @serdes
    @dataclass(frozen=True)
    class UserFrozen:
        name: str
        age: int

    user = UserFrozen(name="Alice", age=20)
    json_data = user.to_json()

    # シリアライズの確認
    assert json_data == '{"name": "Alice", "age": 20}'

    # デシリアライズの確認
    deserialized_user = UserFrozen.from_json(json_data)
    assert deserialized_user == user  # frozen dataclassなのでインスタンス比較が可能


def test_UserNotFrozen_serialize_deserialize():
    @serdes
    @dataclass(frozen=False)
    class UserNotFrozen:
        name: str
        age: int

    user = UserNotFrozen(name="Alice", age=20)
    json_data = user.to_json()

    # シリアライズの確認
    assert json_data == '{"name": "Alice", "age": 20}'

    # デシリアライズの確認
    deserialized_user = UserNotFrozen.from_json(json_data)
    assert deserialized_user.name == user.name
    assert deserialized_user.age == user.age


def test_has_class_user_serialize_deserialize():
    @serdes
    @dataclass(frozen=True)
    class Address:
        value: str

    @serdes
    @dataclass(frozen=True)
    class Email:
        address: Address

    @serdes
    @dataclass(frozen=False)
    class HasEmailUser:
        name: str
        age: int
        email: Email

    user = HasEmailUser(
        name="Alice", age=20, email=Email(address=Address(value="mail@mail.com"))
    )
    json_data = user.to_json()

    # シリアライズの確認
    assert (
        json_data
        == '{"name": "Alice", "age": 20, "email": {"address": {"value": "mail@mail.com"}}}'
    )

    # デシリアライズの確認
    deserialized_user = HasEmailUser.from_json(json_data)
    assert deserialized_user.name == user.name
    assert deserialized_user.age == user.age
    assert deserialized_user.email.address.value == user.email.address.value

```

- /app/deno-sample/doll/engine/core/src/doll_core/serdes.py
```python
import json
from dataclasses import fields, is_dataclass

# to_dict, from_dictあたりはカスタム変換メソッドが使えたほうがいいよなあ

def serdes(cls):
    """
    ### `serdes` デコレータ

    `serdes`は、データクラスにシリアライズおよびデシリアライズ機能を自動的に追加するデコレータです。これを使用することで、データクラスのインスタンスを簡単にJSON形式に変換したり、逆にJSONからオブジェクトに復元することができます。

    ---

    ### ドキュメンテーション

    #### 概要
    `serdes`デコレータは、Pythonのデータクラスに対して以下の2つの主要な機能を追加します。
    1. **シリアライズ（serialize）**: データクラスのインスタンスをJSON文字列に変換します。
    2. **デシリアライズ（deserialize）**: JSON文字列からデータクラスのインスタンスを復元します。

    このデコレータは、ネストされたデータクラスにも対応しており、入れ子構造を持つオブジェクトも正しくシリアライズ・デシリアライズできます。

    #### 追加されるメソッド
    - `to_dict(self)`: データクラスのインスタンスを辞書形式に変換します。
    - `to_json(self)`: データクラスのインスタンスをJSON文字列に変換します。
    - `from_dict(cls, data: dict)`: 辞書形式のデータからデータクラスのインスタンスを生成します。
    - `from_json(cls, json_data: str)`: JSON文字列からデータクラスのインスタンスを生成します。

    ---

    ### 使い方（Usage）

    以下に、`serdes`を使ってデータクラスをシリアライズ・デシリアライズする例を示します。

    #### シンプルなデータクラスの例

    ```python
    from dataclasses import dataclass
    from doll_core.serdes import serdes

    @serdes
    @dataclass
    class User:
        name: str
        age: int

    # インスタンスの作成
    user = User(name="Alice", age=20)

    # シリアライズ: オブジェクトをJSONに変換
    json_data = user.to_json()
    print(json_data)  # '{"name": "Alice", "age": 20}'

    # デシリアライズ: JSONからオブジェクトに復元
    restored_user = User.from_json(json_data)
    print(restored_user)  # User(name='Alice', age=20)
    ```

    #### ネストされたデータクラスの例

    ```python
    @serdes
    @dataclass
    class Address:
        city: str
        country: str

    @serdes
    @dataclass
    class User:
        name: str
        age: int
        address: Address

    # インスタンスの作成
    user = User(name="Alice", age=20, address=Address(city="Tokyo", country="Japan"))

    # シリアライズ: ネストされたオブジェクトをJSONに変換
    json_data = user.to_json()
    print(json_data)
    # '{"name": "Alice", "age": 20, "address": {"city": "Tokyo", "country": "Japan"}}'

    # デシリアライズ: JSONからオブジェクトに復元
    restored_user = User.from_json(json_data)
    print(restored_user)
    # User(name='Alice', age=20, address=Address(city='Tokyo', country='Japan'))
    ```

    ### まとめ

    この`serdes`デコレータは、データクラスのシリアライズ・デシリアライズを簡素化し、ネストされた構造でも扱えるよう設計されています。コードベースが拡大するにつれて、カスタムフィールドの処理や異なるフォーマットの対応など、柔軟に拡張していくことが可能です。
    """

    def to_dict(self) -> dict:
        # クラスのフィールドを取得して辞書形式に変換
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def to_json(self) -> str:
        # クラスのフィールドを取得して辞書形式に変換
        def _to_json(data: dict):
            for key, value in data.items():
                if hasattr(value, "to_dict") and hasattr(value, "to_json"):
                    data[key] = _to_json(value.to_dict())
                else:
                    print("not class key: ", key)
                    data[key] = value
            return data

        data = _to_json(self.to_dict())

        return json.dumps(data)

    @classmethod
    def from_dict(cls, data: dict):
        # データからフィールドにマッピング
        field_names = {f.name for f in fields(cls)}
        kwargs = {key: value for key, value in data.items() if key in field_names}

        # ネストされたオブジェクトのデシリアライズ処理
        for field in fields(cls):
            if is_dataclass(field.type) and field.name in kwargs:
                kwargs[field.name] = field.type.from_dict(kwargs[field.name])

        return cls(**kwargs)

    @classmethod
    def from_json(cls, json_data: str):
        # JSON文字列を辞書に変換
        data = json.loads(json_data)
        return cls.from_dict(data)

    # シリアライズメソッドを追加
    cls.to_dict = to_dict
    cls.to_json = to_json
    cls.from_dict = from_dict
    cls.from_json = from_json
    return cls

```

- /app/deno-sample/doll/engine/core/src/doll_core/value_objects/company_uuid.py
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class CompanyUUID:
    """
    immutableなデータクラスです。
    why: 非同期で処理した際に予期せぬ挙動を防ぐために、このクラスはimmutableです。
    このクラスは、会社のUUIDを表すクラスです。会社固有のIDとして利用します。

    CompanyUUIDはあらかじめ決定的な値です。UUIDは一意である必要があり、一つの会社に対して一つのCompanyUUIDが与えられます。
    """

    __slots__ = "_value"
    _value: str

    def __new__(cls, _value: str) -> "CompanyUUID":
        obj = super().__new__(cls)

        if not _value:
            raise ValueError("CompanyUUID value is required.")

        # UUID形式に合っているかを簡単にチェック
        # TODO: 本当にUUID形式に合っているかをチェックする場合は、以下のコメントアウトを外してください。
        # 現状UUID形式だったか怪しい。。
        # if len(_value) != 36 or _value.count('-') != 4:
        #     raise ValueError(f"{_value} is not a valid UUID format.")

        object.__setattr__(obj, "_value", str(_value))
        return obj

    def __eq__(self, other) -> bool:
        if not isinstance(other, CompanyUUID):
            raise ValueError(f"{other} is not an instance of CompanyUUID.")
        return self._value == other._value

    def __str__(self) -> str:
        return self._value


if __name__ == "__main__":
    company_uuid = "123e4567-e89b-12d3-a456-426614174000"
    company_key = CompanyUUID(company_uuid)

    print(str(company_key))

```

- /app/deno-sample/doll/engine/core/src/doll_core/value_objects/s3_object_key.py
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class S3ObjectKey:
    """
    immutableなデータクラスです。
    why: 非同期で処理した際に予期せぬ挙動を防ぐために、このクラスはimmutableです。
    このクラスは、オブジェクトに関連する一意のキーを表すクラスです。

    ObjectKeyはあらかじめ決定的な値であり、s3のオブジェクトに対して一つのObjectKeyが割り当てられます。
    """

    __slots__ = "_value"
    _value: str

    def __new__(cls, _value: str) -> "S3ObjectKey":
        obj = super().__new__(cls)

        if not _value:
            raise ValueError("ObjectKey value is required.")
        
        # 例えば、S3ObjectKeyが英数字かつハイフンやアンダースコアも許容する場合の検証
        if not _value.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"{_value} is not a valid ObjectKey format. It should only contain alphanumeric characters, dashes, or underscores.")

        object.__setattr__(obj, "_value", str(_value))
        return obj

    def __eq__(self, other) -> bool:
        if not isinstance(other, S3ObjectKey):
            raise ValueError(f"{other} is not an instance of ObjectKey.")
        return self._value == other._value

    def __str__(self) -> str:
        return self._value


if __name__ == "__main__":
    object_key = "OBJECT_KEY_123-456"
    obj_key = S3ObjectKey(object_key)

    print(str(obj_key))

```

- /app/deno-sample/doll/engine/core/src/doll_core/value_objects/input_event_dict.py
```python
from typing import Any
from dataclasses import dataclass


@dataclass(frozen=True)
class InputEventDict:
    """
    入力イベントを受け取るクラスです。
    辞書として振る舞います。
    """

    __slots__ = "_value"
    _value: dict

    def __new__(cls, v: Any) -> "InputEventDict":
        v: dict = cls._check(v)

        obj = super().__new__(cls)
        object.__setattr__(obj, "_value", v)
        return obj

    def __init__(self, v: Any):
        pass

    @staticmethod
    def _check(v: Any) -> dict:
        """
        イベントが辞書型であるかチェックします。
        :param event: Any
        :return: dict
        """

        if v is None:
            raise ValueError("event is required")
        if isinstance(v, dict):
            return v
        else:
            raise ValueError("event is not dict.")

    def __getitem__(self, key: str):
        """
        イベントから指定のキーを取得します。
        :param key: str
        :return: Any
        """
        return self._value[key]

    def get(self):
        """
        イベントを取得します。
        :return: dict
        """
        return self._value

    def __contains__(self, key: str) -> bool:
        """
        指定のキーが存在するか確認します。
        :param key: str
        :return: bool
        """
        return key in self._value

    def keys(self):
        """
        キーの一覧を返します。
        :return: dict_keys
        """
        return self._value.keys()

    def items(self):
        """
        キーと値のペアを返します。
        :return: dict_items
        """
        return self._value.items()

    def values(self):
        """
        値の一覧を返します。
        :return: dict_values
        """
        return self._value.values()


if __name__ == "__main__":
    e = InputEventDict({"key": "value", "another_key": 123})
    print(e["key"])  # "value"を表示
    print("key" in e)  # Trueを表示
    print(e.keys())  # dict_keys(['key', 'another_key'])
    print(e.items())  # dict_items([('key', 'value'), ('another_key', 123)])
    print(e.values())  # dict_values(['value', 123])

    try:
        e = InputEventDict("str")
    except ValueError as e:
        print(e)

```

- /app/deno-sample/doll/engine/core/src/doll_core/value_objects/__init__.py
```python

```

- /app/deno-sample/doll/engine/core/src/doll_core/value_objects/state_key.py
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class StateKey:
    """
    immutableなデータクラスです。
    why: 非同期で処理した際に予期せぬ挙動を防ぐために、このクラスはimmutableです。
    このクラスは、StateのKeyを表すクラスです。state storeに対してcrudを行う際に使用します。

    StateKeyはランダムな値ではなくあらかじめ決定的な値です。
    一つの役割に対して一つのStateKeyを持つことになります。
    
    VistIDをStateKeyとして利用しています。
    """

    # __dict__を生やさないようにしてメモリ消費を抑えるために、__slots__を使っている
    # value onjectは本質的にたくさん生成される可能性が高いです。そのため、__slots__を使ってメモリ消費を抑えることが重要です。
    __slots__ = "_value"
    # __slots__を使うとdefault_factoryが使えなくなります。
    _value: str

    def __new__(cls, _value: str) -> "StateKey":
        obj = super().__new__(cls)

        if not _value:
            raise ValueError("StateKey value is required.")

        # インスタンス作成時に一度だけ作成するため、__setattr__を使って代入する
        object.__setattr__(obj, "_value", str(_value))
        return obj

    def __eq__(self, other) -> bool:
        # pythonだからunreachableなことないんだけどな。。。
        # タイプヒントつけるとpylanceがraiseに到達しないとか言ってくるからこういう書き方になってしまう
        if not isinstance(other, StateKey):
            raise ValueError(f"{other} is not an instance of SpeakerId.")
        return self._value == other._value

    def __str__(self) -> str:
        return self._value


if __name__ == "__main__":
    visit_id = "test"
    state_key = StateKey(visit_id)

    print(str(state_key))

```

- /app/deno-sample/doll/engine/core/src/doll_core/di.py
```python
from typing import Any, List, Tuple, Callable, TypeVar, Union
import injector

Interface = TypeVar("Interface")
Implementataion = Union[Callable[..., Any], Any]


class DIContainer:
    """
    DIコンテナです。
    InterfaceとImplementataionを登録し、commitします。
    Interfaceを引数に取る関数を呼び出すことで、Implementataionのインスタンスを取得できます。
    """

    def __init__(self):
        self.classs: List[Tuple[Interface, Implementataion]] = []

    def register(self, interface: Interface, implementation: Implementataion) -> None:
        self.classs.append((interface, implementation))

    def configure(self, binder: injector.Binder) -> None:
        for interface, implementation in self.classs:
            binder.bind(interface, to=implementation, scope=injector.singleton)

    def commit(self) -> "DIContainer":
        """
        InterfaceとImplementataionを登録し、commitします。
        :return: DIContainer
        """
        self._injector = injector.Injector(self.configure)
        return self

    def __getitem__(self, interface: Interface) -> Implementataion:
        """
        Interfaceを引数に取る関数を呼び出すことで、Implementataionのインスタンスを取得できます。
        ex.) DependencyContainer[Interface]() のように実体クラスを取得する
        :param interface: Interface
        :return: Implementataion
        """
        return lambda: self._injector.get(interface)


if __name__ == "__main__":
    # 依存性として使うクラスを定義
    from abc import ABC, abstractmethod

    class IHelloService(ABC):
        @abstractmethod
        def greet(self):
            pass

        pass

    class HelloService(IHelloService):
        def greet(self):
            return "Hello from HelloService!"

    container = DIContainer()
    # クラスのインスタンスを登録
    container.register(interface=IHelloService, implementation=HelloService)

    dic = container.commit()

    # 関数の引数は固定
    def main(input, context):
        # 関数内で依存性を解決
        hello_service: IHelloService = dic[IHelloService]()

        # 実際の処理
        print(hello_service.greet())

    # サンプルのinputとcontext
    main("sample input", "sample context")

```

- /app/deno-sample/doll/engine/core/src/doll_core/__init__.py
```python

```

- /app/deno-sample/doll/engine/core/src/doll_core/logger.py
```python
import logging
from dataclasses import dataclass


@dataclass(frozen=False)
class Logger:
    __slots__ = ["logger"]
    logger: logging.Logger

    @staticmethod
    def init(class_name: str):
        # 　TODO: ログレベルを環境変数から取得するように変更
        def _min_log_level():
            return logging.DEBUG

        logger = logging.getLogger(class_name)
        logger.setLevel(_min_log_level())

        # コンソールハンドラの作成
        console_handler = logging.StreamHandler()
        console_handler.setLevel(_min_log_level())
        # ログフォーマットの作成
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # ハンドラをロガーに追加
        logger.addHandler(console_handler)

        instance = Logger(logger)

        return instance.logger


if __name__ == "__main__":
    logger = Logger.init(__name__)
    import json

    # サンプルログ出力
    logger.debug("This is a debug message")
    logger.debug(f"my event {json.dumps({"key1": "value1", "key2": "value2"})}")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

```

- /app/deno-sample/doll/engine/core/src/doll_core/aws_s3.py
```python

CHK_RAW_FILE_S3_BUCKET = "publicapi-chk-doll-raw-fil"
```

- /app/deno-sample/doll/engine/identifier/tests/service/test_compute_embedding_service.py
```python
"""
直接実行

rye run pytest -v -s tests/service/test_compute_embedding_service.py::<関数名>

ex)
rye run pytest -v -s tests/service/test_compute_embedding_service.py::test_clustering
"""

import pytest
from components.di import di
from service.compute_service import IComputeService
from components.state_store.abstract import IDiarizerStateStore
from components.utils import get_file_path, load_audio
from components.entity.diarized_speaker import DiarizedSpeaker
from usecase.abstract_usecase import ISpeakerIdentiicationUsecase
import uuid
from const.speaker import SpeakerDataDict
from typing import List
from components.value_objects.speaker_source import SpeakerSource
from components.value_objects.state_key import StateKey


def create_speakers(section_dir) -> List[DiarizedSpeaker]:
    # dirからfileのパスを取得する
    file_paths = get_file_path(section_dir)

    # torchAudioで読み込む
    section_audio_data = [load_audio(file_path=file_path) for file_path in file_paths]

    new_speakers = [
        DiarizedSpeaker(SpeakerSource(value=waveform, sample_rate=sr).ref())
        for waveform, sr in section_audio_data
    ]

    return new_speakers


@pytest.fixture
def setup_and_cleanup():
    # setup
    dic = di.inject()
    usecase: ISpeakerIdentiicationUsecase = dic[ISpeakerIdentiicationUsecase]()

    key = StateKey(str(uuid.uuid4()))

    section1_spekers = create_speakers("/app/sample_audio/section1")
    print("section1_spekers: ", section1_spekers)

    # Add speakers to store init
    usecase.pull_state(section1_spekers, key)

    service: IComputeService = dic[IComputeService]()
    diarizer_store: IDiarizerStateStore = dic[IDiarizerStateStore]()

    section2_spekers = create_speakers("/app/sample_audio/section2")

    # データを同期
    speakers = usecase.pull_state(section2_spekers, key)

    resource = {
        "service": service,
        "diarizer_store": diarizer_store,
        "key": key,
        "speakers": speakers,
    }
    print("Setup: Resource created")
    return resource


def cleanup(func):
    def wrapper(setup_and_cleanup):
        resource = func(setup_and_cleanup)
        key = resource["key"]
        diarizer_store: IDiarizerStateStore = resource["diarizer_store"]
        diarizer_store.remove(key)
        print("Cleanup: Resource destroyed")

    return wrapper


@cleanup
def test_clustering(setup_and_cleanup):
    # 距離データが欲しいので、compute_embeddingを呼び出す
    resource = setup_and_cleanup
    service: IComputeService = resource["service"]
    speakers: SpeakerDataDict = resource["speakers"]
    distances = service.compute_speakers_distance(speakers)
    i_speakers = speakers["identified_speakers"]

    # クラスタリングを実行
    service.clustering(i_speakers, distances)
    return resource

```

- /app/deno-sample/doll/engine/identifier/tests/usecase/test_speaker_identification_usecase.py
```python
"""
rye run pytest -v -s tests/usecase/test_speaker_identification_usecase.py::test_sync_state_initial
"""

import pytest
from components.di import di
from usecase.abstract_usecase import ISpeakerIdentiicationUsecase
from components.state_store.abstract import IDiarizerStateStore
import uuid
import torch
from components.entity.diarized_speaker import DiarizedSpeaker
from typing import List
from components.value_objects.speaker_source import SpeakerSource
from components.value_objects.state_key import StateKey


@pytest.fixture
def setup_and_cleanup():
    # setup
    dic = di.inject()
    usecase: ISpeakerIdentiicationUsecase = dic[ISpeakerIdentiicationUsecase]()
    diarizer_store: IDiarizerStateStore = dic[IDiarizerStateStore]()
    resource = {"usecase": usecase, "diarizer_store": diarizer_store}
    print("Setup: Resource created")
    return resource


def cleanup(func):
    def wrapper(setup_and_cleanup):
        resource = func(setup_and_cleanup)
        key = resource["key"]
        diarizer_store: IDiarizerStateStore = resource["diarizer_store"]
        diarizer_store.remove(key)
        print("Cleanup: Resource destroyed")

    return wrapper


@cleanup
def test_sync_state_initial(setup_and_cleanup):
    resource = setup_and_cleanup
    usecase: ISpeakerIdentiicationUsecase = resource["usecase"]
    key = StateKey(str(uuid.uuid4()))
    new_speakers = [
        DiarizedSpeaker(SpeakerSource(value=torch.rand(1, 50000), sample_rate=16000)),
        DiarizedSpeaker(SpeakerSource(value=torch.rand(1, 50000), sample_rate=16000)),
        DiarizedSpeaker(SpeakerSource(value=torch.rand(1, 50000), sample_rate=16000)),
    ]

    speakers = usecase.pull_state(new_speakers, key)
    assert speakers is None

    resource["key"] = key

    return resource


@cleanup
def test_sync_state_exists_speakers(setup_and_cleanup):
    resource = setup_and_cleanup
    usecase: ISpeakerIdentiicationUsecase = resource["usecase"]
    key = StateKey(str(uuid.uuid4()))
    new_speakers = [
        DiarizedSpeaker(SpeakerSource(value=torch.rand(1, 50000), sample_rate=16000)),
        DiarizedSpeaker(SpeakerSource(value=torch.rand(1, 50000), sample_rate=16000)),
        DiarizedSpeaker(SpeakerSource(value=torch.rand(1, 50000), sample_rate=16000)),
    ]

    # Add speakers to store
    speakers = usecase.pull_state(new_speakers, key)
    assert speakers is None

    new_speakers = [
        DiarizedSpeaker(SpeakerSource(value=torch.rand(1, 50000), sample_rate=16000)),
        DiarizedSpeaker(SpeakerSource(value=torch.rand(1, 50000), sample_rate=16000)),
        DiarizedSpeaker(SpeakerSource(value=torch.rand(1, 50000), sample_rate=16000)),
    ]
    # Sync speakers
    speakers = usecase.pull_state(new_speakers, key)

    assert speakers is not None

    resource["key"] = key

    return resource

```

- /app/deno-sample/doll/engine/identifier/lambda_function.py
```python
import os

# pytonでwhooami
print("whoami: ", os.system("whoami"))

cwd = os.getcwd()
print("cwd: ", cwd)
import sys

sys.path.append(os.path.join(cwd, "src"))
sys.path.append(os.path.join(cwd, ".venv", "lib", "python3.12", "site-packages"))
from src.app.identify_handler import handler as app_handler


def handler(event, context):
    return app_handler(event, context)

```

- /app/deno-sample/doll/engine/identifier/tool/multiple-audio-identification-sample/app.py
```python
# 1. 圧縮ファイルを解凍する(解凍したファイルは後で削除する)

import sys

sys.path.append("/app")
sys.path.append("/app/src")

from tool.mock_dializer import mock_dializer

import tarfile
import os

from components.value_objects.state_key import StateKey
from src.app.identify_handler import handler
from components.logger import Logger
import logging
import time

import json

# 削除したいファイルのパス
files_to_delete = [
    "/app/identifier/identifier-dev.log",
    "/app/tool/multiple-audio-identification-sample/multiple-audio-identification-sample.log",
]

# ファイルを削除する
for file_path in files_to_delete:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

logger = Logger.init(f"{__file__}:{__name__}")
file_handler = logging.FileHandler(
    "/app/tool/multiple-audio-identification-sample/multiple-audio-identification-sample.log"
)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# 解凍する.tarファイルのパス
tar_file = "/app/tool/multiple-audio-identification-sample/archive_diarized_speakers_data_expected_to_5.tar.xz"

# 解凍先のディレクトリ
output_dir = "/app/tool/multiple-audio-identification-sample/tmp"

# 解凍処理
if not os.path.exists(os.path.join(output_dir, "section_0")):
    with tarfile.open(tar_file, "r:xz") as tar:
        tar.extractall(path=output_dir)

print(f"解凍が完了しました: {output_dir}")

# 2. mock_dializer.pyを実行して、ダミーデータを生成する

sections = [
    os.path.join(output_dir, root)
    for root in sorted(os.listdir(output_dir))
    if root.startswith("section_")
]


for i, try_section_dir in enumerate(sections):
    company_uuid = "fjdskfghrusjk3842734824627846fhdjskfbnsdkfhbsk"
    destination = "dest"
    visit_id = "shirakamifubukikonkonkitsune123456"
    object_keys = [
        StateKey(
            f"companies/{company_uuid}/visits/{i}/test/1720742400_1720746000/{item}"
        )
        for item in os.listdir(try_section_dir)
    ]

    # mock_dializerを実行する

    body = mock_dializer(
        try_section_dir=try_section_dir,
        company_uuid=company_uuid,
        Destination=destination,
        visit_id=visit_id,
        object_keys=object_keys,
    )

    logger.info(f"追加したobject {i}回目 {len(object_keys)}個")

    # bodyをidentifierに投げる
    logger.info(f"話者識別を開始します。 {i}回目")
    start = time.perf_counter()
    start_ms = start * 1000  # ミリ秒に変換
    logger.info(f"start handler: {start:.6f} seconds ({start_ms:.2f} ms)")
    logger.debug(f"request body: {json.dumps(body)}")
    result = handler(body, None)
    logger.info(
        f"successed to identifier process: {time.perf_counter() - start:.6f} seconds, {time.perf_counter() * 1000 - start_ms:.2f} ms"
    )

    logger.info(f"result: {json.dumps(result)}\n----------")

```

- /app/deno-sample/doll/engine/identifier/tool/mock_dializer.py
```python
"""
本番では使用しません。
ローカルで動作確認のためにだけ使用します。

実際に話者分離を行うのではなく、すでに話者分離済みの音源(/app/sample_audio/*)を使用します。
こちらを環境変数で指定されてるstoreに保存します。
"""

import os
from components.utils import get_file_path, load_audio
from components.entity.diarized_speaker import DiarizedSpeaker
from components.value_objects.speaker_source import SpeakerSource
from components.entity.diarized_speakers import DiarizedSpeakers
import sys
from components.di import di
from components.state_store.abstract import IDiarizerStateStore
from components.value_objects.state_key import StateKey
import uuid


def mock_dializer(
    try_section_dir: str,
    company_uuid: str,
    destination: str,
    visit_id: str = "1234",
    object_keys: list = [],
) -> dict:
    # directoryか確認する
    if not os.path.isdir(try_section_dir):
        raise f"{try_section_dir} is not a directory."

    dic = di.inject()
    state_store: IDiarizerStateStore = dic[IDiarizerStateStore]()

    # dirからfileのパスを取得する
    file_paths = get_file_path(try_section_dir)

    # torchAudioで読み込む
    audio_data = [load_audio(file_path=file_path) for file_path in file_paths]
    new_speakers = [
        DiarizedSpeaker(speaker_source=SpeakerSource(value=waveform, sample_rate=sr))
        for waveform, sr in audio_data
    ]

    print("state_store", state_store)

    # モックでやることとしては、ここで保存しておく必要があるわけね
    # -> s3 minioに接続する口を作る

    # 保存します -> keyが返ってくる -> keyを返す

    # keyはspeaker_dialzierに属するので持っててもいいか
    identifier_input = {
        "CompanyUUID": company_uuid,
        "VisitID": visit_id,
        "Destination": destination,
        "ObjectKeys": object_keys,
    }

    for object_key, d_speaker in zip(identifier_input["ObjectKeys"], new_speakers):
        object_key: StateKey = object_key
        d_speaker: DiarizedSpeaker = d_speaker
        state_store.send(object_key, d_speaker.serialize())

    dss = DiarizedSpeakers(new_speakers[0])
    for new_speaker in new_speakers[1:]:
        dss.append(new_speaker)

    state = dss.serialize()
    json_data = state.to_json()
    # シリアライズされたデータのバイトサイズを取得
    size_in_bytes = sys.getsizeof(json_data)

    # 400KBを超えるかどうかをチェック
    print(f"""
          Item size:
          {size_in_bytes} bytes

          {size_in_bytes / 1024} KB

          {size_in_bytes / 1024 / 1024} MB

          {size_in_bytes / 1024 / 1024 / 1024} GB

          """)
    if size_in_bytes > 400 * 1024:  # 400KB = 400 * 1024 bytes
        print(f"❌ Item size is {size_in_bytes} bytes, which exceeds the 400KB limit.")
    else:
        print(
            f"⭕️ Item size is {size_in_bytes} bytes, which is within the 400KB limit."
        )

    # TODO: 本当はdiarizers.seriarize()でやりたいが、今のdiarizerにその実装を入れれてない。。
    # なので後日対応します

    identifier_input["ObjectKeys"] = [
        str(object_key) for object_key in identifier_input["ObjectKeys"]
    ]

    return identifier_input


if __name__ == "__main__":
    import json

    try_section_dir = "/app/identifier/sample_audio/section1"
    # try_section_dir = "/app/sample_audio/section2"

    company_uuid = "4ce9cf31-4bae-4cf5-bc18-55ddb061d024"
    destination = "dest"
    visit_id = "1234"
    object_keys = [
        StateKey(
            f"companies/{company_uuid}/visits/1/test/1720742400_1720746000/SPEAKER_00.wav"
        ),
        StateKey(
            f"companies/{company_uuid}/visits/1/test/1720742400_1720746000/SPEAKER_01.wav"
        ),
    ]
    body = mock_dializer(
        try_section_dir, company_uuid, destination, visit_id, object_keys
    )

    print(f"""登録できたよ！
          コピーして使ってね！

          rye run identify-dev-local '{json.dumps(body)}'
          
          """)

```

- /app/deno-sample/doll/engine/identifier/tool/create_test_bucket_data.py
```python
import boto3

from components.env import Env
from components.value_objects.bucket import BucketDestDiarizedAudio


env = Env()

AWS_ACCESS_KEY_ID = env.AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY = env.AWS_SECRET_ACCESS_KEY
AWS_ENDPOINT_URL = env.AWS_S3_ENDPOINT_URL
print(AWS_ENDPOINT_URL)
AWS_REGION = env.AWS_REGION_NAME

AUDIO_PATH = "/app/identifier/sample_audio/chunk0.wav"
AUDIO_B64_PATH = "/app/identifier/sample_audio/chunk0_b64.wav"
SRC_BUCKET = str(BucketDestDiarizedAudio())
UPLOAD_OBJECT_KEY = "companies/4ce9cf31-4bae-4cf5-bc18-55ddb061d024/visits/1/1720742400_1720746000.wav.b64"


s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    endpoint_url=AWS_ENDPOINT_URL,
)

s3.create_bucket(Bucket=SRC_BUCKET)

print("create test bucket data successed")

```

- /app/deno-sample/doll/engine/identifier/src/repository/__init__.py
```python

```

- /app/deno-sample/doll/engine/identifier/src/app/identify_handler.py
```python
import json
import sys

sys.path.append("/app/src")
sys.path.append("/app/src/app")

import time
from components.di import di
from components.logger import Logger
from components.value_objects.state_key import StateKey
from usecase.abstract_usecase import ISpeakerIdentiicationUsecase


dic = di.inject()
logger = Logger.init(f"{__file__}:{__name__}")


# step functionsではこう言うRequestで来るらしい
# TODO:
# {
#     "ExecutedVersion": "$LATEST",
#     "Payload": {
#         "CompanyUUID": "4ce9cf31-4bae-4cf5-bc18-55ddb061d024",
#         "VisitID": "1234",
#         "Destination": "publicapi-chk-doll-encoded-file",
#         "ObjectKeys": [
#             "companies/4ce9cf31-4bae-4cf5-bc18-55ddb061d024/visits/1/test/1720742400_1720746000/007d601ee65a4f44ac0e67a9b76b68f2.wav",
#             "companies/4ce9cf31-4bae-4cf5-bc18-55ddb061d024/visits/1/test/1720742400_1720746000/155591b55f55482b9eec6ed406ebb165.wav"
#         ]
#     },
#     "SdkHttpMetadata": {
#         "AllHttpHeaders": {
#             "X-Amz-Executed-Version": [
#                 "$LATEST"
#             ],
#             "x-amzn-Remapped-Content-Length": [
#                 "0"
#             ],
#             "Connection": [
#                 "kee
# ...
def handler(input_event, context):
    start = time.perf_counter()
    start_ms = start * 1000  # ミリ秒に変換
    logger.info(f"start handler: {start:.6f} seconds ({start_ms:.2f} ms)")

    # TODO: value objectsで行う
    logger.debug(f"event type: {type(input_event)}")
    if input_event is None:
        raise ValueError("event is required")
    if isinstance(input_event, dict):
        logger.debug(f"event: {json.dumps(input_event)}, context: {type(context)}")
    else:
        logger.warning(f"event: {json.dumps(input_event)}, context: {type(context)}")
        raise TypeError("event must be a dict")

    logger.debug(f"event: {json.dumps(input_event)}")

    if "Payload" not in input_event:
        raise ValueError("Payload is not found in event.")

    input_event = input_event["Payload"]

    if "VisitID" not in input_event:
        raise ValueError("VisitID is not found in event.")
    identifier_key = StateKey(input_event["VisitID"])

    input_event["VisitID"] = identifier_key

    # usecaseの取得
    usecase: ISpeakerIdentiicationUsecase = dic[ISpeakerIdentiicationUsecase]()

    # データの同期[dializer stateのpull]
    new_speakers = usecase.pull_dialized_speaker(input_event)

    logger.debug(f"new_speakers size: {len(new_speakers)}")

    if new_speakers is None:
        raise ValueError("new_speakers is None.")

    # データの同期[identifier stateのpull]
    speaker_data = usecase.pull_state(
        new_speakers=new_speakers,
        identifier_key=identifier_key,
    )

    # 何もないならデータの登録だけされて抜ける
    if speaker_data is None:
        logger.info("speaker_data is None.")
        return

    # speaker_dataがある場合は、speaker_dataを使って処理を行う
    identified_speakers = usecase.identify(speaker_data)

    # データの保存[stateのpush]
    usecase.push_state(
        identifier_key=identifier_key, identified_speakers=identified_speakers
    )

    logger.info(
        f"successed to identifier process: {time.perf_counter() - start:.6f} seconds, {time.perf_counter() * 1000 - start_ms:.2f} ms"
    )
    logger.info(f"identified_speakers size: {identified_speakers.size}")

    return {
        "statusCode": 200,
        "identifierdSpeakersSize": identified_speakers.size,
    }

```

- /app/deno-sample/doll/engine/identifier/src/app/__init__.py
```python

```

- /app/deno-sample/doll/engine/identifier/src/app/identify_dev_local.py
```python
import sys
from components.di import di
from usecase.abstract_usecase import ISpeakerIdentiicationUsecase
from components.value_objects.state_key import StateKey
import json
from components.logger import Logger

# DIコンテナの取得
dic = di.inject()
logger = Logger.init(f"{__file__}:{__name__}")


# script実行起点
# 音源分離された生のデータを使って起動する
def identify_dev_local():
    args = sys.argv
    if len(args) < 1:
        logger.error("Error: 引数が足りません。")
        return

    event = json.loads(args[1])
    logger.debug(f"event: {json.dumps(event)}")
    # identifier speakersの管理に使う
    #TODO: value objectsで行う
    if "VisitID" not in event:
        raise ValueError("VisitID is not found in event.")
    identifier_key = StateKey(event["VisitID"])

    event["VisitID"] = identifier_key

    # usecaseの取得
    usecase: ISpeakerIdentiicationUsecase = dic[ISpeakerIdentiicationUsecase]()

    # データの同期[dializer stateのpull]
    new_speakers = usecase.pull_dialized_speaker(event)

    if new_speakers is None:
        raise ValueError("new_speakers is None.")

    # データの同期[identifier stateのpull]
    speaker_data = usecase.pull_state(
        new_speakers=new_speakers,
        identifier_key=identifier_key,
    )

    # 何もないならデータの登録だけされて抜ける
    if speaker_data is None:
        logger.info("speaker_data is None.")
        return

    # speaker_dataがある場合は、speaker_dataを使って処理を行う
    identified_speakers = usecase.identify(speaker_data)

    # データの保存[stateのpush]
    usecase.push_state(
        identifier_key=identifier_key, identified_speakers=identified_speakers
    )

    serialized_item = identified_speakers.to_json()

    with open("identified_speakers.json", "w") as f:
        f.write(serialized_item)

    # シリアライズされたデータのバイトサイズを取得
    size_in_bytes = sys.getsizeof(serialized_item)

    # 400KBを超えるかどうかをチェック
    if size_in_bytes > 400 * 1024:  # 400KB = 400 * 1024 bytes
        logger.info(
            f"❌Item size is {size_in_bytes} bytes, which exceeds the 400KB limit."
        )
    else:
        logger.info(
            f"⭕️Item size is {size_in_bytes} bytes, which is within the 400KB limit."
        )

    # 400kBは全然こえなさそう

```

- /app/deno-sample/doll/engine/identifier/src/const/__init__.py
```python
import io
from typing import List, Tuple, TypedDict, Dict
import torch
from enum import Enum


DirPath = str
DiarizedWavPath = str
SampleRate = int


class PersonalAudioDict(TypedDict):
    waveform: torch.Tensor
    sample_rate: int


class StateStoreType(Enum):
    REDIS = "REDIS"
    S3 = "S3"
    DYNAMODB = "DYNAMODB"
    S3_MINIO = "S3_MINIO"


# 類似性の閾値
SIMILARITY_THRESHOLD = 0.83
# この値は実験的にこれくらいが良さそうだったので、この値を採用しています。

```

- /app/deno-sample/doll/engine/identifier/src/const/speaker.py
```python
from components.value_objects.speaker_id import SpeakerId
from typing import List, Tuple, TypedDict, Dict
from components.abstracts import ISpeaker, IDiarizedSpeaker, IIdentifiedSpeakers


IdentifiedSpeakersDict = Dict[SpeakerId, ISpeaker]


class SpeakerDataDict(TypedDict):
    new_speakers: List[IDiarizedSpeaker]
    identified_speakers: IIdentifiedSpeakers

```

- /app/deno-sample/doll/engine/identifier/src/components/speaker_identification/abstract.py
```python
from abc import ABC, abstractmethod
from components.abstracts import ISpeaker


class ICompute(ABC):
    @abstractmethod
    def compute_distance(self, speaker: ISpeaker) -> ISpeaker:
        pass

```

- /app/deno-sample/doll/engine/identifier/src/components/speaker_identification/embedding.py
```python
# TODO : constにおく, dializerと共通で定義したいな、、
OFFLINE_MODEL_PATH = "/root/.cache/pyannote_cache"

from pyannote.audio import Inference
import const
from pyannote.audio import Model
import numpy as np

from components.speaker_identification.abstract import ICompute
from components.abstracts import ISpeaker
from components.value_objects.embedding_vector import EmbeddingVector
import threading


class ComputeEmbedding(ICompute):
    def __init__(self):
        model = Model.from_pretrained(
            "pyannote/embedding", cache_dir=OFFLINE_MODEL_PATH
        )
        self.inference = Inference(model, window="whole")

    def compute_distance(self, speaker: ISpeaker) -> ISpeaker:
        result = self.inference(speaker.memory)
        speaker.embedding_vector = EmbeddingVector(result)
        return speaker

```

- /app/deno-sample/doll/engine/identifier/src/components/speaker_identification/__init__.py
```python

```

- /app/deno-sample/doll/engine/identifier/src/components/speaker_identification/vo.py
```python
import numpy as np
from typing import List, Tuple, TypedDict


class _DistanceMatrix(TypedDict):
    distance: float
    x: np.ndarray
    y: np.ndarray


class DistanceMatrix:
    # なぜ初期値が1.0なのか、疑問に思うかもしれません。
    # コサイン距離は0から1の間を取るため、もっとも離れた値である1.0を初期値に設定しています。
    def __init__(self, x: np.ndarray, y: np.ndarray, distance: float = 1.0):
        self.matrix: _DistanceMatrix = {"distance": distance, "x": x, "y": y}

    def value(self) -> _DistanceMatrix:
        return self.matrix

```

- /app/deno-sample/doll/engine/identifier/src/components/speaker_identification/identify.py
```python


```

- /app/deno-sample/doll/engine/identifier/src/components/di/di.py
```python
"""
serverlessの仕様的に一回しか実行されないはずなので、globalでdiを保持しておく
"""

from components import di
from components.speaker_identification import abstract, embedding
from service import abstract_service, compute_service
from usecase import abstract_usecase, speaker_identification_usecase
import const
from components.utils import get_state_store_service, STATE_STORE_SERVICE_VALUE
from components import env
from components.state_store.abstract import (
    IIdentifierStateStore,
    IDiarizerStateStore,
    IStateStore,
)
from typing import Protocol

container = di.DIContainer()

container.register(interface=env.IEnv, implementation=env.Env)


container.register(
    interface=abstract.ICompute, implementation=embedding.ComputeEmbedding
)

container.register(
    interface=abstract_service.IComputeService,
    implementation=compute_service.ComputeService,
)

container.register(
    interface=abstract_usecase.ISpeakerIdentiicationUsecase,
    implementation=speaker_identification_usecase.SpeakerIdentificationUsecase,
)


class StateStore(Protocol):
    IIdentifierStateStore
    IDiarizerStateStore


def set_state_store(
    conteiner: di.DIContainer,
    value: STATE_STORE_SERVICE_VALUE,
    interface: IStateStore,
) -> di.DIContainer:
    state_store_service = get_state_store_service(value)

    if const.StateStoreType.REDIS == state_store_service:
        from components.state_store.store_redis import StoreRedis

        container.register(interface=interface, implementation=StoreRedis)

    elif const.StateStoreType.S3_MINIO == state_store_service:
        from components.state_store.store_s3 import StoreS3

        container.register(interface=interface, implementation=StoreS3)
    elif const.StateStoreType.S3 == state_store_service:
        from components.state_store.store_s3 import StoreS3

        container.register(interface=interface, implementation=StoreS3)
    elif const.StateStoreType.DYNAMODB == state_store_service:
        from components.state_store.store_dynamo import StoreDynamo

        container.register(interface=interface, implementation=StoreDynamo)
    # TODO dynamo dbの追加
    else:
        raise ValueError(f"Invalid state store type. {state_store_service}")

    return conteiner


# 環境変数の有無によって異なる
# dializerが分離した音源を保持しているstateStore
conteiner = set_state_store(
    container, "DIARIZER_STATE_STORE_SERVICE", IDiarizerStateStore
)
# identifierが識別したスピーカーを保持しているstateStore
conteiner = set_state_store(
    conteiner, "IDENTIFIER_STATE_STORE_SERVICE", IIdentifierStateStore
)


def inject() -> di.DIContainer:
    return container()


if __name__ == "__main__":
    dic = inject()
    e1 = dic[abstract.ICompute]()

    e2 = dic[abstract.ICompute]()

    assert e1 is e2

    e3 = dic[IDiarizerStateStore]()
    e4 = dic[IIdentifierStateStore]()

```

- /app/deno-sample/doll/engine/identifier/src/components/di/__init__.py
```python
from typing import Any, List, Tuple, Callable, TypeVar, Union
import injector


Interface = TypeVar("Interface")
Implementataion = Union[Callable[..., Any], Any]


class DIContainer:
    def __init__(self):
        self.classs: List[Tuple[Interface, Implementataion]] = []

    def register(self, interface: Interface, implementation: Implementataion) -> None:
        self.classs.append((interface, implementation))

    def configure(self, binder: injector.Binder) -> None:
        for interface, implementation in self.classs:
            binder.bind(interface, to=implementation, scope=injector.singleton)

    def __call__(self) -> "DIContainer":
        self._injector = injector.Injector(self.configure)
        return self

    def __getitem__(self, interface: Interface) -> Implementataion:
        # ex.) DependencyContainer[Interface]() のように実体クラスを取得する
        return lambda: self._injector.get(interface)

```

- /app/deno-sample/doll/engine/identifier/src/components/logger/__init__.py
```python
import logging
from dataclasses import dataclass
import os


@dataclass(frozen=False)
class Logger:
    __slots__ = ["logger"]
    logger: logging.Logger

    @staticmethod
    def init(class_name: str):
        # 　TODO: ログレベルを環境変数から取得するように変更
        def _min_log_level():
            return logging.DEBUG

        logger = logging.getLogger(class_name)
        logger.setLevel(_min_log_level())

        # コンソールハンドラの作成
        console_handler = logging.StreamHandler()
        console_handler.setLevel(_min_log_level())
        # ログフォーマットの作成
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # 環境変数からファイル出力するかどうかを取得する
        log_file_path = os.environ.get("OUTPUT_LOG_FILE")

        if log_file_path is not None:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(_min_log_level())
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # ハンドラをロガーに追加
        logger.addHandler(console_handler)

        instance = Logger(logger)

        return instance.logger


if __name__ == "__main__":
    logger = Logger.init(__name__)
    import json

    # サンプルログ出力
    logger.debug("This is a debug message")
    logger.debug(f"my event {json.dumps({"key1": "value1", "key2": "value2"})}")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

```

- /app/deno-sample/doll/engine/identifier/src/components/entity/identified_speakers.py
```python
from components.value_objects.speaker_id import SpeakerId
from components.abstracts import IIdentifiedSpeakers, IIdentifiedSpeaker, IState
from typing import Dict
from components.value_objects.state import State
from components.entity.identified_speaker import IdentifiedSpeaker
from components.value_objects.state_key import StateKey


# di入れた方がと思ったけどだめ
# 他のリクエストが入ってきて他の会議の値が入ったりするとカオスになる。
class IdentifiedSpeakers(IIdentifiedSpeakers):
    # entityも外部から暗黙的なプロパティ追加されることは想定しないので__slotsを定義する
    __slots__ = ("_data", "_size", "_state_key")
    _data: Dict[SpeakerId, IIdentifiedSpeaker]
    _size: int
    _state_key: StateKey

    def __init__(
        self, key: SpeakerId, value: IIdentifiedSpeaker, state_key: StateKey
    ) -> None:
        self._data = {key: value}
        self._size = 1
        self._state_key = state_key
        pass

    def add(self, key: SpeakerId, value: IIdentifiedSpeaker) -> None:
        self._data[key] = value
        self._size += 1

    @property
    def data(self) -> Dict[SpeakerId, IIdentifiedSpeaker]:
        return self._data

    @property
    def size(self) -> int:
        return self._size

    @property
    def state_key(self) -> StateKey:
        return self._state_key

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, key: SpeakerId) -> IIdentifiedSpeaker:
        return self._data[key]

    def serialize(self) -> State:
        data = {
            str(key): value.serialize().to_dict() for key, value in self._data.items()
        }
        data["state_key"] = str(self._state_key)

        # ここでデータのサイズを保存しておく
        # データサイズは、ミーティングの終了時とかに表示するのに使う
        data["identified_speaker_size"] = self._size
        data = State(key=self.__class__.__name__, value=data)
        return data

    @staticmethod
    def deserialize(state: IState) -> "IdentifiedSpeakers":
        # TODO: かなり冗長に書いてるのでリファクタリングする
        if state.key != IdentifiedSpeakers.__name__:
            raise ValueError(f"Expected {IdentifiedSpeakers.__name__}, got {state.key}")

        value = state.value[IdentifiedSpeakers.__name__]

        value_keys = list(value.keys())

        # state_keyは省く
        value_keys = [
            key
            for key in value_keys
            if key not in {"state_key", "identified_speaker_size"}
        ]

        state_key = StateKey(value["state_key"])

        first_key = value_keys[0]
        first_value = value[first_key]
        key_class_name: str = list(first_value.keys())[0]
        first_value = first_value[key_class_name]
        first_speaker_id = SpeakerId(first_key)
        identified_speaker = IdentifiedSpeaker.deserialize(
            State(key=key_class_name, value=first_value), speaker_id=first_speaker_id
        )

        instance = IdentifiedSpeakers(
            key=first_speaker_id, value=identified_speaker, state_key=state_key
        )

        for key in value_keys[1:]:
            inner_value = value[key]
            speaker_id = SpeakerId(key)
            inner_key_class_name: str = list(inner_value.keys())[0]
            class_value = inner_value[inner_key_class_name]
            identified_speaker = IdentifiedSpeaker.deserialize(
                State(key=inner_key_class_name, value=class_value),
                speaker_id=speaker_id,
            )
            instance.add(speaker_id, identified_speaker)

        return instance

    def to_dict(self) -> Dict:
        raise NotImplementedError

    def to_json(self) -> str:
        return self.serialize().to_json()

```

- /app/deno-sample/doll/engine/identifier/src/components/entity/diarized_speakers.py
```python
from components.abstracts import IDiarizedSpeakers, IDiarizedSpeaker, IState
from typing import List, Dict
from components.value_objects.state import State
from typing import List, Iterator
from components.entity.diarized_speaker import DiarizedSpeaker


class DiarizedSpeakers(IDiarizedSpeakers):
    """
    Listとして振る舞うDiarizedSpeakerの集合を表すクラス。
    dializerと共通。
    """

    # entityも外部から暗黙的なプロパティ追加されることは想定しないので__slotsを定義する
    __slots__ = "_array"
    _array: List[IDiarizedSpeaker]

    def __init__(self, value: IDiarizedSpeaker) -> None:
        self._array = [value]
        pass

    def append(self, value: IDiarizedSpeaker) -> None:
        self._array.append(value)

    @property
    def size(self) -> int:
        return len(self._array)

    def __len__(self) -> int:
        return len(self._array)

    def serialize(self) -> State:
        data = [value.serialize().to_dict() for value in self._array]

        data = State(key=self.__class__.__name__, value=data)
        return data

    @staticmethod
    def deserialize(state: IState) -> "IDiarizedSpeakers":
        # TODO: かなり冗長に書いてるのでリファクタリングする
        if state.key != DiarizedSpeakers.__name__:
            raise ValueError(f"Expected {DiarizedSpeakers.__name__}, got {state.key}")

        values: List = state.value

        first_value = values[0]
        key_class_name: str = list(first_value.keys())[0]
        value_dializer = first_value[key_class_name]

        dializer_speaker = DiarizedSpeaker.deserialize(
            State(key=key_class_name, value=value_dializer)
        )

        dializer_speakers = DiarizedSpeakers(dializer_speaker)

        for value in values[1:]:
            key_class_name: str = list(value.keys())[0]
            value_dializer = value[key_class_name]
            dializer_speaker = DiarizedSpeaker.deserialize(
                State(key=key_class_name, value=value_dializer)
            )

            dializer_speakers.append(dializer_speaker)

        return dializer_speakers

    def to_dict(self) -> None:
        raise NotImplementedError

    def to_json(self) -> str:
        return self.serialize().to_json()

    # 以下はlistとして振る舞うために必要
    def __getitem__(self, index: int) -> IDiarizedSpeaker:
        return self._array[index]

    def __len__(self) -> int:
        return len(self._array)

    def __iter__(self) -> Iterator[IDiarizedSpeaker]:
        return iter(self._array)

    def __contains__(self, item: IDiarizedSpeaker) -> bool:
        return item in self._array

    @property
    def array(self) -> List[IDiarizedSpeaker]:
        return self._array


if __name__ == "__main__":
    import os
    from components.utils import get_file_path, load_audio
    from components.entity.diarized_speaker import DiarizedSpeaker
    from components.value_objects.speaker_source import SpeakerSource
    import sys
    import json

    try_section_dir = "/app/sample_audio/section1"

    if not os.path.isdir(try_section_dir):
        raise f"{try_section_dir} is not a directory."

    # dirからfileのパスを取得する
    file_paths = get_file_path(try_section_dir)

    # torchAudioで読み込む
    audio_data = [load_audio(file_path=file_path) for file_path in file_paths]
    new_speakers = [
        DiarizedSpeaker(speaker_source=SpeakerSource(value=waveform, sample_rate=sr))
        for waveform, sr in audio_data
    ]

    dss = DiarizedSpeakers(new_speakers[0])
    for new_speaker in new_speakers[1:]:
        dss.append(new_speaker)

    json_data = dss.to_json()

    # シリアライズされたデータのバイトサイズを取得
    size_in_bytes = sys.getsizeof(json_data)

    # 400KBを超えるかどうかをチェック
    print(f"""
          Item size: 
          {size_in_bytes} bytes

          {size_in_bytes / 1024} KB
          
          {size_in_bytes / 1024 / 1024} MB
          
          {size_in_bytes / 1024 / 1024 / 1024} GB
          
          """)
    if size_in_bytes > 400 * 1024:  # 400KB = 400 * 1024 bytes
        print(f"❌ Item size is {size_in_bytes} bytes, which exceeds the 400KB limit.")
    else:
        print(
            f"⭕️ Item size is {size_in_bytes} bytes, which is within the 400KB limit."
        )

    with open("dialized_speakers.json", "w") as f:
        f.write(json_data)

    with open("dialized_speakers.json", "r") as f:
        data = f.read()

    response_dict: Dict = json.loads(data)

    key_class_name = list(response_dict.keys())[0]

    state = State(key=key_class_name, value=response_dict[key_class_name])

    a = DiarizedSpeakers.deserialize(state)

```

- /app/deno-sample/doll/engine/identifier/src/components/entity/diarized_speaker.py
```python
import const
from components.value_objects.embedding_vector import EmbeddingVector
from components.abstracts import IDiarizedSpeaker
import copy
from components.value_objects.speaker_source import SpeakerSource
from typing import Optional
from components.value_objects.state import State
from typing import Dict
from components.abstracts import ISerializable
from components.value_objects.speaker_id import SpeakerId


class DiarizedSpeaker(IDiarizedSpeaker):
    """
    音源分離されたスピーカーの情報を保持するクラス
    """

    __slots__ = ("_speaker_source", "_embedding_vector", "_tmp_speaker_id")

    # cloneして新しいaddressを作成したいのは_embedding_vectorだけでwaveformは同じaddressを使いたい。。。

    def __init__(self, speaker_source: Optional[SpeakerSource] = None):
        self._speaker_source: Optional[SpeakerSource] = speaker_source

        # 埋め込みベクトルを求める以外の方法になるかもしれない
        self._embedding_vector: Optional[EmbeddingVector] = None

        # 一時的にspeaker_idを保持する変数
        # 話者識別されるまでの一時的な変数
        self._tmp_speaker_id = SpeakerId()

    def __eq__(self, other: "DiarizedSpeaker") -> bool:
        if not isinstance(other, DiarizedSpeaker):
            return False
        return self._tmp_speaker_id == other._tmp_speaker_id

    @property
    def speaker_source(self) -> Optional[SpeakerSource]:
        return getattr(self, "_speaker_source", None)

    def serialize(self) -> State:
        key = self.__class__.__name__

        if self.embedding_vector is None and self.speaker_source is not None:
            value = {
                "_speaker_source": self.speaker_source.serialize().to_dict(),
                "_embedding_vector": None,
            }
        elif self.embedding_vector is not None:
            value = {
                "_speaker_source": None,
                "_embedding_vector": self.embedding_vector.serialize().to_dict(),
            }
        else:
            raise Exception("speaker_source and embedding_vector are None.")
        state = State(key=key, value=value)
        return state

    @staticmethod
    def deserialize(state: ISerializable) -> "DiarizedSpeaker":
        class_name_from_state = state.key
        if class_name_from_state != DiarizedSpeaker.__name__:
            raise Exception("class name is not matched.")
        # TODO: これは良くないけど、、
        state_value = state.value
        if set(state.value.keys()) == {"_speaker_source", "_embedding_vector"}:
            state_value = {DiarizedSpeaker.__name__: state.value}

        embedding_vector_value: Dict = state_value[class_name_from_state][
            "_embedding_vector"
        ]

        speaker_source_value: Dict = state_value[class_name_from_state][
            "_speaker_source"
        ]

        ds = DiarizedSpeaker()
        if embedding_vector_value is not None:
            class_name = EmbeddingVector.__name__
            embedding_vector = EmbeddingVector.deserialize(
                State(key=class_name, value=embedding_vector_value[class_name])
            )
            ds.embedding_vector = embedding_vector
        if speaker_source_value is not None:
            class_name = SpeakerSource.__name__
            speaker_source = SpeakerSource.deserialize(
                State(key=class_name, value=speaker_source_value[class_name])
            )
            ds._speaker_source = speaker_source

        return ds

    def to_dict(self) -> None:
        raise NotImplementedError

    def to_json(self) -> str:
        raise NotImplementedError

    @property
    def memory(self):
        if self.speaker_source is None:
            raise Exception("speaker_source is not set.")
        return const.PersonalAudioDict(
            waveform=self.speaker_source.tensor,
            sample_rate=self.speaker_source.sample_rate,
        )

    @property
    def embedding_vector(self) -> Optional[EmbeddingVector]:
        return self._embedding_vector

    @embedding_vector.setter
    def embedding_vector(self, a: EmbeddingVector) -> Optional[EmbeddingVector]:
        # 埋め込みベクトルを計算したら、speaker_sourceは不要になるので削除する
        # 後続の処理でspeaker_sourceが必要になることはなく、埋め込みベクトル同士の比較を行なっていくため。

        if self.speaker_source is not None:
            del self._speaker_source
        self._embedding_vector = a

    def clone(self) -> "DiarizedSpeaker":
        """
        # この関数は、speakerの情報をコピーして新しいspeakerを作成する
        # 非同期で処理した際に、pythonは参照渡しを行うため、speakerの情報が上書きされてしまう
        # この時、予期せぬ挙動を防ぐために、この関数を用いてspeakerの情報をコピーする
        """
        res = copy.deepcopy(self)
        return res

```

- /app/deno-sample/doll/engine/identifier/src/components/entity/identified_speaker.py
```python
from components.value_objects.speaker_id import SpeakerId
from components.abstracts import IIdentifiedSpeaker, IDiarizedSpeaker, ISerializable
from components.value_objects.embedding_vector import EmbeddingVector
from components.entity.diarized_speaker import DiarizedSpeaker
from components.value_objects.state import State
from components.value_objects.speakers_distance import SpeakersDistance
import copy
from typing import List


class IdentifiedSpeaker(IIdentifiedSpeaker):
    # entityも外部から暗黙的なプロパティ追加されることは想定しないので__slotsを定義する
    __slots__ = ("_typical_speaker", "_speaker_id", "_distances")

    def __init__(self, first_speaker: IDiarizedSpeaker, speaker_id: SpeakerId):
        # 比較元のspeaker。
        # もっとも特徴量を表すspeakerにするか、データを増やしてもっとも特徴量を表すspeakerにするかは要検討
        self._typical_speaker: IDiarizedSpeaker = first_speaker
        # speaker label
        self._speaker_id: SpeakerId = speaker_id

        self._distances: List[SpeakersDistance] = []

    def __eq__(self, other: "IdentifiedSpeaker") -> bool:
        if not isinstance(other, IdentifiedSpeaker):
            return False
        return self._speaker_id == other.speaker_id

    def serialize(self, to_dict: bool = False) -> State:
        """
        この関数は、speakerの情報をシリアライズする

        """
        key = self.__class__.__name__
        value = {
            "speaker_id": str(self._speaker_id),
            "typical_speaker": self._typical_speaker.serialize().to_dict(),
        }

        state = State(key=key, value=value)

        return state

    @property
    def distances(self) -> List[SpeakersDistance]:
        return self._distances

    def add_distances(self, distances: SpeakersDistance) -> None:
        # TODO: 新しく追加された距離の方がより特徴量を表しているなら、typical_speakerを更新する
        # もしくは、平均を取るなり、足して圧縮するなりする。
        # より精度が良さそうになりそうな方法を選択する

        # そのためには、埋め込みベクトルの計算元のデータを保持しておくとか改善の余地あり
        self._distances.append(distances)

        # typical_speakerを更新する
        # 音声が途中で切れたとしても、会話の終わりと会話の始まりは同一人物なら似ている確率が高そう
        # 時系列が保証されてた方がいいわけね
        self._typical_speaker = distances.new_speaker

    @staticmethod
    def deserialize(state: ISerializable, speaker_id: SpeakerId) -> "IdentifiedSpeaker":
        # speaker_idは、識別された話者のラベルです。
        # 辞書のキーとしてこのインスタンスを使用しています。
        # これと同じ値を渡せていないと、検索できなくて死にます
        # ※pythonの辞書がたのキーにインスタンスを渡した場合、アドレス値が異なると異なるキーとして扱われます

        class_name_from_state = state.key
        if class_name_from_state != IdentifiedSpeaker.__name__:
            raise Exception("class name is not matched.")
        value = state.value
        typical_speaker_name = list(value["typical_speaker"].keys())[0]
        typical_speaker_value = value["typical_speaker"][typical_speaker_name]
        state = State(key=typical_speaker_name, value=typical_speaker_value)
        typical_speaker = DiarizedSpeaker.deserialize(state)

        instance = IdentifiedSpeaker(typical_speaker, speaker_id)

        return instance

    def to_dict(self) -> None:
        raise NotImplementedError

    def to_json(self) -> str:
        raise NotImplementedError

    @property
    def speaker_id(self) -> SpeakerId:
        return self._speaker_id

    @property
    def memory(self):
        return self._typical_speaker.memory

    @property
    def embedding_vector(self) -> EmbeddingVector:
        return self._typical_speaker.embedding_vector

    @embedding_vector.setter
    def embedding_vector(self, a: EmbeddingVector) -> None:
        self._typical_speaker.embedding_vector = a

    def clone(self) -> "IdentifiedSpeaker":
        """
        # この関数は、speakerの情報をコピーして新しいspeakerを作成する
        # 非同期で処理した際に、pythonは参照渡しを行うため、speakerの情報が上書きされてしまう
        # この時、予期せぬ挙動を防ぐために、この関数を用いてspeakerの情報をコピーする
        """
        return copy.deepcopy(self)

```

- /app/deno-sample/doll/engine/identifier/src/components/value_objects/speaker_id.py
```python
from dataclasses import dataclass
import uuid
from typing import Optional


@dataclass(frozen=True)
class SpeakerId:
    """
    immutableなデータクラスです。
    why: 非同期で処理した際に予期せぬ挙動を防ぐために、このクラスはimmutableです。
    このクラスは、speakerIDを表すクラスです。
    インスタンスが呼ばれた際に、speakerIDを一度だけ生成します。
    """

    # __dict__を生やさないようにしてメモリ消費を抑えるために、__slots__を使っている
    # value onjectは本質的にたくさん生成される可能性が高いです。そのため、__slots__を使ってメモリ消費を抑えることが重要です。
    __slots__ = "_speaker_id"
    # __slots__を使うとdefault_factoryが使えなくなります。
    _speaker_id: str

    def __new__(cls, speaker_id: Optional[str] = None) -> "SpeakerId":
        obj = super().__new__(cls)
        # ここでspeakerIDを生成する
        if not speaker_id:
            speaker_id = str(uuid.uuid4().hex)
        # インスタンス作成時に一度だけ作成するため、__setattr__を使って代入する
        object.__setattr__(obj, "_speaker_id", speaker_id)
        return obj

    # @dataclass(frozen=True)したときに__init__が自動的に作成されるが、そちらは勝手に引数を期待してしまうので、
    # それを上書きして無効にするために書いています
    def __init__(self, speaker_id: Optional[str] = None):
        pass

    def __eq__(self, other) -> bool:
        # pythonだからunreachableなことないんだけどな。。。
        # タイプヒントつけるとpylanceがraiseに到達しないとか言ってくるからこういう書き方になってしまう
        if not isinstance(other, SpeakerId):
            raise ValueError(f"{other} is not an instance of SpeakerId.")
        return self._speaker_id == other._speaker_id

    def __str__(self) -> str:
        return self._speaker_id


if __name__ == "__main__":
    speaker_id = SpeakerId()
    print(str(speaker_id))
    print(str(speaker_id))
    print(str(speaker_id))

    speaker_id = SpeakerId()
    print(str(speaker_id))

    speaker_id = SpeakerId()
    print(str(speaker_id))

    id = "test"
    speaker_id = SpeakerId(id)
    print(str(speaker_id))

```

- /app/deno-sample/doll/engine/identifier/src/components/value_objects/bucket.py
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class BucketDestDiarizedAudio:
    """
    immutableなデータクラスです。
    why: 非同期で処理した際に予期せぬ挙動を防ぐために、このクラスはimmutableです。
    """

    # __dict__を生やさないようにしてメモリ消費を抑えるために、__slots__を使っている
    # value onjectは本質的にたくさん生成される可能性が高いです。そのため、__slots__を使ってメモリ消費を抑えることが重要です。
    __slots__ = "_value"
    # __slots__を使うとdefault_factoryが使えなくなります。
    _value: str

    def __new__(cls) -> "BucketDestDiarizedAudio":
        obj = super().__new__(cls)

        # 固定値
        _value = "publicapi-chk-doll-encoded-file"

        # インスタンス作成時に一度だけ作成するため、__setattr__を使って代入する
        object.__setattr__(obj, "_value", _value)
        return obj

    def __init__(self) -> None:
        pass

    def __eq__(self, other) -> bool:
        # pythonだからunreachableなことないんだけどな。。。
        # タイプヒントつけるとpylanceがraiseに到達しないとか言ってくるからこういう書き方になってしまう
        if not isinstance(other, BucketDestDiarizedAudio):
            raise ValueError(f"{other} is not an instance of BucketDestDiarizedAudio.")
        return self._value == other._value

    def __str__(self) -> str:
        return self._value


if __name__ == "__main__":
    visit_id = "test"
    a = BucketDestDiarizedAudio()

    print(str(a))

```

- /app/deno-sample/doll/engine/identifier/src/components/value_objects/s3_object_key.py
```python
from dataclasses import dataclass
from components.value_objects.state_key import StateKey


@dataclass(frozen=True)
class S3ObjectKey:
    """
    immutableなデータクラスです。
    why: 非同期で処理した際に予期せぬ挙動を防ぐために、このクラスはimmutableです。
    このクラスは、StateのKeyを表すクラスです。state storeに対してcrudを行う際に使用します。
    """

    # __dict__を生やさないようにしてメモリ消費を抑えるために、__slots__を使っている
    # value onjectは本質的にたくさん生成される可能性が高いです。そのため、__slots__を使ってメモリ消費を抑えることが重要です。
    __slots__ = "_value"
    # __slots__を使うとdefault_factoryが使えなくなります。
    _value: StateKey

    def __new__(cls, _value: StateKey) -> "S3ObjectKey":
        obj = super().__new__(cls)

        if not _value:
            raise ValueError("S3ObjectKey value is required.")

        if not isinstance(_value, StateKey):
            raise ValueError(f"{_value} is not an instance of StateKey.")

        # インスタンス作成時に一度だけ作成するため、__setattr__を使って代入する
        object.__setattr__(obj, "_value", _value)
        return obj

    def __eq__(self, other) -> bool:
        # pythonだからunreachableなことないんだけどな。。。
        # タイプヒントつけるとpylanceがraiseに到達しないとか言ってくるからこういう書き方になってしまう
        if not isinstance(other, S3ObjectKey):
            raise ValueError(f"{other} is not an instance of S3ObjectKey.")
        return self._value == other._value

    def __str__(self) -> str:
        return self._value._value


if __name__ == "__main__":
    visit_id = "test"
    state_key = S3ObjectKey(StateKey(visit_id))

    print(str(state_key))

```

- /app/deno-sample/doll/engine/identifier/src/components/value_objects/speaker_source.py
```python
from dataclasses import dataclass
import numpy as np
from torch import Tensor
from components.abstracts import IRefSerializable
import gzip
import weakref
from components.value_objects.state import State
from components.abstracts import IState
from typing import Dict
import torch
from components.logger import Logger
from logging import Logger as LoggerType


@dataclass(frozen=True)
class SpeakerSource(IRefSerializable):
    """

    immutableなデータクラスです。
    why: 非同期で処理した際に予期せぬ挙動を防ぐために、このクラスはimmutableです。

    このクラスは、話者分離アルゴリズムによって分離された音声データをカプセル化します。
    音声データは、torch.tensor または np.ndarray , Noneとして提供され、用途に応じて柔軟に扱うことができます。

    また、弱参照で参照するため、メモリ消費を抑えることができます。

    """

    # __dict__を生やさないようにしてメモリ消費を抑えるために、__slots__を使っている
    # value onjectは本質的にたくさん生成される可能性が高いです。そのため、__slots__を使ってメモリ消費を抑えることが重要です。
    __slots__ = ("_value", "_sample_rate", "_logger")
    _value: np.ndarray | Tensor
    _sample_rate: int
    _logger: LoggerType

    def __new__(cls, value: np.ndarray | Tensor, sample_rate: int) -> "SpeakerSource":
        obj = super().__new__(cls)
        logger = Logger.init(f"{__file__}:{__name__}")
        # インスタンス作成時に一度だけ作成するため、__setattr__を使って代入する
        object.__setattr__(obj, "_value", value)
        object.__setattr__(obj, "_sample_rate", sample_rate)
        object.__setattr__(obj, "_logger", logger)

        return obj

    # @dataclass(frozen=True)したときに__init__が自動的に作成されるが、そちらは勝手に引数を期待してしまうので、
    # それを上書きして無効にするために書いています
    def __init__(self, value: np.ndarray | Tensor, sample_rate: int) -> None:
        pass

    def ref(self) -> "SpeakerSource":
        """
        弱参照を返す。
        弱参照は、参照カウントを増やさずにオブジェクトを参照することができる。

        戻り値が"SpeakerSource"となっていますが嘘です。本当は"weakref.ReferenceType"です。
        なぜこうしているのかというとtype hintingが殺されてしまうからです。。。
        くそうpythonめ。。。

        弱参照使うとパフォーマンス追い求めれますが予期せぬガベージコレクションが発生する可能性があるので注意してください。。
        c/c++でありがちなやつ、最近(2024/08/15周辺)だとクラウドストライクのあれです。(めっちゃおもろいので興味ある人は調べてね。)
        rustならこういう問題起きないんだけどな。。。
        rustのスマートポインタが恋しい。。

        merit:
        参照カウンタが増えないため、メモリを節約できる。
        また、del hoge みたいにすることで、オブジェクトを削除扱い(※諸説あり。元々占有してたメモリをほぼほぼ解放するからまあ。。。)にすることができる。

        demerit:
        参照カウンタが増えないため、予期せぬタイミングでガベージコレクションが発生する可能性がある。
        例えば、予期せぬタイミングで参照カウンタが0になったときにガベージコレクションが発生することがある。

        why:
        話者分離アルゴリズムによって分離された音声データのテンソルはそもそも大きなオブジェクトである。
        それがずっと保持されるということは線形、最悪には指数的にメモリを消費することになる。
        具体的には、会議が数時間と長丁場になったとすると、その分生じるテンソルの数も増えるし、カウントされる話者が一人増えるたびに組み合わせがn通り増えることになる。
        これをずっと保持し続けると、メモリを圧迫することになり、プログラムが予期せぬクラッシュをしたり、よくわからないスケールをしたりする可能性がある。
        これを防ぐために、弱参照を使って、参照カウンタを増やさずオブジェクトを参照することができるようにしている。
        """
        rfr = weakref.ref(self)
        return rfr()

    def __del__(self):
        self._logger.info(f"SpeakerSource:{id(self)} is deleted.")

    @property
    def value(self) -> np.ndarray | Tensor:
        return self._value

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def ndarray(self) -> np.ndarray:
        if isinstance(self._value, np.ndarray):
            return self._value
        raise ValueError("not np.ndarray")

    @property
    def tensor(self) -> Tensor:
        if isinstance(self._value, Tensor):
            return self._value
        raise ValueError("not Tensor")

    def serialize(self) -> IState:
        """
        SpeakerSourceをシリアライズする。
        重いから頻繁に叩かないこと。
        """
        key = self.__class__.__name__

        if isinstance(self._value, Tensor):
            data = self._value.numpy().tobytes()
        else:
            data = self._value.tobytes()

        gzip_data = gzip.compress(data).hex()
        data = {
            "sample_rate": self._sample_rate,
            "shape": self._value.shape,
            "dtype": str(self._value.dtype),
            "data": gzip_data,
            "is_tensor": isinstance(self._value, Tensor),
            "is_ndarray": isinstance(self._value, np.ndarray),
        }

        state = State(key=key, value=data)
        return state

    @staticmethod
    def deserialize(state: IState) -> "SpeakerSource":
        # np.ndarray | Tensorとして解釈可能かどうかをチェックする
        class_name_from_state = state.key
        if class_name_from_state != SpeakerSource.__name__:
            raise Exception("class name is not matched.")

        state_value: Dict = state.value
        sample_rate = state_value["sample_rate"]

        gzipped_value = bytes.fromhex(state_value["data"])
        decompressed_value = gzip.decompress(gzipped_value)
        # state_value["dtype"] に対応する torch のデータ型を取得
        dtype_str = state_value["dtype"]
        dtype: np.float32 = None
        if dtype_str == "torch.float32":
            dtype = np.float32
        is_tensor = state_value["is_tensor"]
        if is_tensor:
            shape = tuple(state_value["shape"])
            # dtype_str を適切な torch のデータ型に変換
            value = torch.from_numpy(
                np.frombuffer(decompressed_value, dtype=dtype).reshape(shape)
            )
            return SpeakerSource(value, sample_rate)

        is_ndarray = state_value["is_ndarray"]
        if is_ndarray:
            shape = tuple(state_value["shape"])
            dtype = np.dtype(state_value["dtype"])
            value = np.frombuffer(decompressed_value, dtype=dtype).reshape(shape)
            return SpeakerSource(value, sample_rate)

    def to_dict(self) -> None:
        raise NotImplementedError

    def to_json(self) -> str:
        raise NotImplementedError


if __name__ == "__main__":

    def print_size(obj):
        from pympler import asizeof

        size_in_bytes = asizeof.asizeof(obj)
        size_in_gb = size_in_bytes / (1024**3)
        return f"Size in GB: {size_in_gb:.10f} GB"

    sr = 16000

    ss = SpeakerSource(
        np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), sr
    ).ref()

    print(ss.serialize())

    state = ss.serialize()

    ss.deserialize(state)

```

- /app/deno-sample/doll/engine/identifier/src/components/value_objects/embedding_vector.py
```python
import numpy as np
from dataclasses import dataclass
from components.abstracts import ISerializable
import gzip
import json
from components.value_objects.state import State
from typing import Dict


@dataclass(frozen=True)
class EmbeddingVector(ISerializable):
    """
    埋め込みベクトルを表すクラスです。
    """

    # TODO: 計算元の音声データのサイズを保持しておく。
    # それを保持しておくことで、最も特徴量を表す話者を選択する際に、使用する。

    __slots__ = "value"
    value: np.ndarray

    def __new__(cls, value: np.ndarray) -> "EmbeddingVector":
        obj = super().__new__(cls)
        cls._check(value)
        object.__setattr__(obj, "value", value)
        return obj

    @staticmethod
    def _check(value: np.ndarray):
        """
        形状は(512,)であることを確認します。
        """
        if value.shape != (512,):
            raise ValueError(
                f"The shape of the embedding vector must be (512,), but got {value.shape}"
            )

    def serialize(self) -> State:
        key = self.__class__.__name__

        value = self.value.tobytes()
        dtype = str(self.value.dtype)
        gzipped_value = gzip.compress(value).hex()
        data = {
            "value": gzipped_value,
            "dtype": dtype,
            "shape": self.value.shape,
        }

        state = State(key=key, value=data)

        return state

    @staticmethod
    def deserialize(state: ISerializable) -> "EmbeddingVector":
        class_name_from_state = state.key
        if class_name_from_state != EmbeddingVector.__name__:
            raise Exception("class name is not matched.")

        state_value: Dict = state.value
        gzipped_value = bytes.fromhex(state_value["value"])
        decompressed_value = gzip.decompress(gzipped_value)
        # TODO: 検証する なぜDecimal???

        shape = state_value["shape"]
        shape = tuple([int(shape[0])])  # 形状を読み込み
        dtype = np.dtype(state_value["dtype"])  # dtypeを読み込み

        value = np.frombuffer(decompressed_value, dtype=dtype).reshape(shape)

        return EmbeddingVector(value=value)

    def to_dict(self) -> None:
        raise NotImplementedError

    def to_json(self) -> str:
        raise NotImplementedError

```

- /app/deno-sample/doll/engine/identifier/src/components/value_objects/__init__.py
```python

```

- /app/deno-sample/doll/engine/identifier/src/components/value_objects/state_key.py
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class StateKey:
    """
    immutableなデータクラスです。
    why: 非同期で処理した際に予期せぬ挙動を防ぐために、このクラスはimmutableです。
    このクラスは、StateのKeyを表すクラスです。state storeに対してcrudを行う際に使用します。

    StateKeyはランダムな値ではなくあらかじめ決定的な値です。
    一つの役割に対して一つのStateKeyを持つことになります。

    """

    # __dict__を生やさないようにしてメモリ消費を抑えるために、__slots__を使っている
    # value onjectは本質的にたくさん生成される可能性が高いです。そのため、__slots__を使ってメモリ消費を抑えることが重要です。
    __slots__ = "_value"
    # __slots__を使うとdefault_factoryが使えなくなります。
    _value: str

    def __new__(cls, _value: str) -> "StateKey":
        obj = super().__new__(cls)

        if not _value:
            raise ValueError("StateKey value is required.")

        # インスタンス作成時に一度だけ作成するため、__setattr__を使って代入する
        object.__setattr__(obj, "_value", _value)
        return obj

    def __eq__(self, other) -> bool:
        # pythonだからunreachableなことないんだけどな。。。
        # タイプヒントつけるとpylanceがraiseに到達しないとか言ってくるからこういう書き方になってしまう
        if not isinstance(other, StateKey):
            raise ValueError(f"{other} is not an instance of SpeakerId.")
        return self._value == other._value

    def __str__(self) -> str:
        return self._value


if __name__ == "__main__":
    visit_id = "test"
    state_key = StateKey(visit_id)

    print(str(state_key))

```

- /app/deno-sample/doll/engine/identifier/src/components/value_objects/sorted_array.py
```python
from dataclasses import dataclass, field
from typing import TypeVar, List, Iterator


T = TypeVar("T")  # ジェネリック型を定義


@dataclass(frozen=True)
# python3.12 以降では class Alice[T]のようにgenericsを使うことができます。
class AscendingSortedArray[T]:
    """
    昇順にソートされた配列であることを保証するイミュータブルなデータクラスです。
    """

    _array: List[T] = field(default_factory=list)

    def __post_init__(self):
        # 初期化後に、リストを降順にソートする処理を行う
        object.__setattr__(self, "_array", sorted(self._array, reverse=False))

    # ダックタイピングについて
    # pythonでは以下のメソッドを持つクラスはイテラブルとして扱われます。
    # ざっくりいうと下記のコードのようにfor文で回したり、indexでアクセスしたり配列として振る舞うことができます。

    # クラスが配列として振る舞うのに必要なメソッドは以下のとおりです。
    # __getitem__
    # __len__
    # __iter__
    # __contains__
    # 詳細についてはググってください。

    def __getitem__(self, index: int) -> T:
        return self._array[index]

    def __len__(self) -> int:
        return len(self._array)

    def __iter__(self) -> Iterator[T]:
        return iter(self._array)

    def __contains__(self, item: T) -> bool:
        return item in self._array

    @property
    def array(self) -> List[T]:
        return self._array


if __name__ == "__main__":
    a = AscendingSortedArray(
        [
            3,
            54,
            1,
            5,
            4,
            5,
            11,
        ]
    )

    print("array", a.array)
    print("len", len(a))

    for i in a:
        print(i)

    # これはエラーになります
    # a._array = 10
    pass

```

- /app/deno-sample/doll/engine/identifier/src/components/value_objects/speakers_distance.py
```python
from __future__ import annotations
from components.abstracts import (
    ISpeaker,
    IIdentifiedSpeaker,
    IDiarizedSpeaker,
    ISpeakersDistance,
)
from typing import Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakersDistance(ISpeakersDistance):
    """
    immutableなデータクラスです。
    why: 非同期で処理した際に予期せぬ挙動を防ぐために、このクラスはimmutableです。
    このクラスは、2つの話者間の距離を表すクラスです。
    このクラスは、2つの話者のうち、少なくとも1つが識別済み話者であることを前提としています。
    """

    # __dict__を生やさないようにしてメモリ消費を抑えるために、__slots__を使っている
    # value onjectは本質的にたくさん生成される可能性が高いです。そのため、__slots__を使ってメモリ消費を抑えることが重要です。
    __slots__ = ("_distance", "_identified_speaker", "_new_speaker")
    _distance: float
    _identified_speaker: IIdentifiedSpeaker
    _new_speaker: IDiarizedSpeaker

    def __new__(
        cls, distance: float, speaker1: ISpeaker, speaker2: ISpeaker
    ) -> "SpeakersDistance":
        obj = super().__new__(cls)
        identified_speaker, new_speaker = cls._check(speaker1, speaker2)
        # インスタンス作成時に一度だけ代入するため、__setattr__を使って代入する
        object.__setattr__(obj, "_distance", distance)
        object.__setattr__(obj, "_identified_speaker", identified_speaker)
        object.__setattr__(obj, "_new_speaker", new_speaker)
        return obj

    def _compare_type_check(self, other: "SpeakersDistance" | float) -> bool:
        return isinstance(other, SpeakersDistance)

    @staticmethod
    def _check(
        speaker1: ISpeaker, speaker2: ISpeaker
    ) -> Tuple[IIdentifiedSpeaker, IDiarizedSpeaker]:
        # 両方とも持つことはあり得ない
        is_speaker1_identified = isinstance(speaker1, IIdentifiedSpeaker)
        is_speaker2_identified = isinstance(speaker2, IIdentifiedSpeaker)

        if is_speaker1_identified and is_speaker2_identified:
            raise ValueError(
                "Both speaker1 and speaker2 are identified speakers, which is not allowed."
            )

        # 両方falseでもあり得ない
        if not is_speaker1_identified and not is_speaker2_identified:
            raise ValueError(
                "Neither speaker1 nor speaker2 is an identified speaker, at least one must be identified."
            )

        # どちらかがidを持っている場合、持っている方を識別済み話者として扱う
        if is_speaker1_identified:
            return speaker1, speaker2
        return speaker2, speaker1

    @property
    def identified_speaker(self) -> IIdentifiedSpeaker:
        return self._identified_speaker

    @property
    def new_speaker(self) -> IDiarizedSpeaker:
        return self._new_speaker

    # ダックタイピングで比較演算子(>, <)を実装
    def __gt__(self, other: "SpeakersDistance" | float):
        if self._compare_type_check(other):
            return self._distance > other._distance
        elif isinstance(other, (int, float)):
            return self._distance > other
        return NotImplemented

    def __lt__(self, other: "SpeakersDistance" | float):
        if self._compare_type_check(other):
            return self._distance < other._distance
        elif isinstance(other, (int, float)):
            return self._distance < other
        return NotImplemented

    def __eq__(self, other: "SpeakersDistance") -> bool:
        return self._distance == other._distance

    def __float__(self) -> float:
        return self._distance


if __name__ == "__main__":
    # 偽のインターフェース定義（実際にはこれらを適切に定義する必要があります）
    class ISpeaker:
        pass

    class IIdentifiedSpeaker(ISpeaker):
        pass

    class IDiarizedSpeaker(ISpeaker):
        pass

    speaker1 = IIdentifiedSpeaker()
    speaker2 = IDiarizedSpeaker()
    speaker3 = IDiarizedSpeaker()

    distance1 = SpeakersDistance(0.5, speaker1, speaker2)
    distance2 = SpeakersDistance(1.0, speaker1, speaker3)
    distance3 = SpeakersDistance(0.2, speaker2, speaker1)

    distances = [distance1, distance2, distance3]

    # 降順にソート
    sorted_distances = sorted(distances, reverse=True)

    for sd in sorted_distances:
        print(sd._distance)

```

- /app/deno-sample/doll/engine/identifier/src/components/value_objects/state.py
```python
from dataclasses import dataclass
import json
from typing import Dict
from components.abstracts import IState


@dataclass(frozen=True)
class State(IState):
    """
    state storageで扱うデータクラスです。
    """

    __slots__ = ("key", "value")
    key: str
    value: Dict

    # why: なぜstateKeyのようなvalueObjectではなくprimitive valueを使っているのか？
    # Stateが引数として受け取るkeyはdeserializeされるclass名であることが大半だからです。
    # Stateはcomponentsであるため、潜在的にどこからでも呼ばれる可能性があり、さまざまな責務を実現するために使用されます。
    # そのため、valueObjectsで縛ってしまうと柔軟性が低下するため、keyはprimitive valueを使っています。
    def __new__(cls, key: str, value: Dict) -> "State":
        obj = super().__new__(cls)
        object.__setattr__(obj, "key", key)
        object.__setattr__(obj, "value", value)
        return obj

    # TODO: Stateは本来、必ずstateKeyを一つ持つことを保証すべき

    # でっかいjsonを扱う時にtupleで扱うとメモリを節約できるかもしれない。
    def to_dict(self) -> Dict:
        return {self.key: self.value}

    def to_json(self) -> str:
        data = self.to_dict()
        return json.dumps(data)

    def serialize(self) -> "State":
        return self

    def deserialize(self, data: str) -> None:
        raise NotImplementedError

```

- /app/deno-sample/doll/engine/identifier/src/components/__init__.py
```python

```

- /app/deno-sample/doll/engine/identifier/src/components/utils/debug_tools.py
```python
def print_size(obj):
    from pympler import asizeof

    size_in_bytes = asizeof.asizeof(obj)
    size_in_gb = size_in_bytes / (1024**3)
    return f"Size in GB: {size_in_gb:.10f} GB"

```

- /app/deno-sample/doll/engine/identifier/src/components/utils/__init__.py
```python
import const
from typing import List, Tuple, Literal
import torchaudio
import torch
import os


def get_file_path(dir_path: const.DirPath) -> List[const.DiarizedWavPath]:
    file_paths: List[const.DiarizedWavPath] = []

    for root, _, files in os.walk(dir_path):
        for filename in files:
            # パスを作成
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


def load_audio(
    file_path: const.DiarizedWavPath,
) -> Tuple[torch.Tensor, const.SampleRate]:
    return torchaudio.load(file_path)


STATE_STORE_SERVICE_VALUE = Literal[
    "DIARIZER_STATE_STORE_SERVICE", "IDENTIFIER_STATE_STORE_SERVICE"
]


def get_state_store_service(
    value: STATE_STORE_SERVICE_VALUE,
) -> const.StateStoreType:
    val = os.getenv(value)
    if val is None:
        raise f"{value} is not defined"

    try:
        return const.StateStoreType(val)
    except ValueError:
        raise f"{value} is invalid. {val}"

```

- /app/deno-sample/doll/engine/identifier/src/components/env/__init__.py
```python
import os
from typing import Protocol
from components.utils import get_state_store_service
import const
from dataclasses import dataclass


class IEnv(Protocol):
    # env
    # TODO: 定数にする
    IDENTIFIER_ENV: str
    # state store type
    DIARIZER_STATE_STORE_SERVICE: const.StateStoreType
    IDENTIFIER_STATE_STORE_SERVICE: const.StateStoreType
    # redis
    REDIS_HOST: str
    REDIS_PORT: str
    REDIS_CHANNNEL: str
    # aws dynamodb
    AWS_REGION_NAME: str
    AWS_DYNAMODB_ENDPOINT_URL: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    # aws s3(minio)
    AWS_S3_ENDPOINT_URL: str


@dataclass(frozen=True)
class Env(IEnv):
    """
    動的にセットされる全ての環境変数を保持するイミュータブルクラス
    """

    # env
    # TODO: 定数にする
    IDENTIFIER_ENV: str
    # state store type
    DIARIZER_STATE_STORE_SERVICE: const.StateStoreType
    IDENTIFIER_STATE_STORE_SERVICE: const.StateStoreType
    # redis
    REDIS_HOST: str
    REDIS_PORT: str
    REDIS_CHANNNEL: str
    # aws dynamodb
    AWS_REGION_NAME: str
    AWS_DYNAMODB_ENDPOINT_URL: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    # aws s3(minio)
    AWS_S3_ENDPOINT_URL: str

    def __new__(cls) -> "Env":
        obj = super().__new__(cls)
        # env
        _identifier_env = os.getenv("IDENTIFIER_ENV") or ""

        # state store type
        _dializer_state_store_service = (
            get_state_store_service("DIARIZER_STATE_STORE_SERVICE") or ""
        )
        if _dializer_state_store_service == "":
            raise ValueError("DIARIZER_STATE_STORE_SERVICE is not set")
        _identifier_state_store_service = (
            get_state_store_service("IDENTIFIER_STATE_STORE_SERVICE") or ""
        )
        if _identifier_state_store_service == "":
            raise ValueError("IDENTIFIER_STATE_STORE_SERVICE is not set")
        # redis
        _redis_host = os.getenv("REDIS_HOST") or ""
        _redis_port = os.getenv("REDIS_PORT") or ""
        _redis_channnel = os.getenv("REDIS_CHANNNEL") or ""
        # dynamo db
        _aws_region_name = os.getenv("AWS_REGION_NAME") or ""
        _aws_dynamodb_endpoint_url = os.getenv("AWS_DYNAMODB_ENDPOINT_URL") or ""
        _aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID") or ""
        _aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY") or ""
        # s3(minio)
        _aws_s3_endpoint_url = os.getenv("AWS_S3_ENDPOINT_URL") or ""

        # env
        object.__setattr__(obj, "IDENTIFIER_ENV", _identifier_env)

        # state store type
        object.__setattr__(
            obj, "DIARIZER_STATE_STORE_SERVICE", _dializer_state_store_service
        )
        object.__setattr__(
            obj, "IDENTIFIER_STATE_STORE_SERVICE", _identifier_state_store_service
        )
        # redis
        object.__setattr__(obj, "REDIS_HOST", _redis_host)
        object.__setattr__(obj, "REDIS_PORT", _redis_port)
        object.__setattr__(obj, "REDIS_CHANNNEL", _redis_channnel)
        # dynamo db
        object.__setattr__(obj, "AWS_REGION_NAME", _aws_region_name)
        object.__setattr__(obj, "AWS_DYNAMODB_ENDPOINT_URL", _aws_dynamodb_endpoint_url)
        object.__setattr__(obj, "AWS_ACCESS_KEY_ID", _aws_access_key_id)
        object.__setattr__(obj, "AWS_SECRET_ACCESS_KEY", _aws_secret_access_key)
        # s3(minio)
        object.__setattr__(obj, "AWS_S3_ENDPOINT_URL", _aws_s3_endpoint_url)

        return obj

    def __init__(self) -> None:
        pass


if __name__ == "__main__":
    env = Env()
    print(env.STATE_STORE_SERVICE)
    print(env.REDIS_HOST)
    pass

```

- /app/deno-sample/doll/engine/identifier/src/components/abstracts/__init__.py
```python
from abc import ABC, abstractmethod
import const
from components.value_objects.speaker_id import SpeakerId
from typing import Any, Optional, Dict, List
from components.value_objects.state_key import StateKey


# シリアライズ可能なクラスのためのインターフェース
class ISerializable(ABC):
    @abstractmethod
    def serialize(self) -> "IState":
        pass

    @staticmethod
    @abstractmethod
    def deserialize(serialized: "IState") -> "IState":
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        pass

    @abstractmethod
    def to_json(self) -> str:
        pass

    # TODO: stateはpropertyとしてvalueを持つべき


class IState(ISerializable):
    pass


class IRefSerializable(ISerializable):
    @abstractmethod
    def ref(self) -> "IRefSerializable":
        """
        弱参照を返す。
        """
        pass

    @abstractmethod
    def __del__(self):
        pass


class ISpeaker(ISerializable):
    @property
    @abstractmethod
    def memory(self) -> "const.PersonalAudioDict":
        pass

    @property
    @abstractmethod
    def embedding_vector(self) -> ISerializable:
        pass

    @embedding_vector.setter
    @abstractmethod
    def embedding_vector(self, a: ISerializable) -> None:
        pass

    @abstractmethod
    def clone(self) -> "ISpeaker":
        pass


class IDiarizedSpeaker(ISpeaker):
    @property
    @abstractmethod
    def speaker_source(self) -> Optional[IRefSerializable]:
        pass


class ISpeakersDistance(ABC):
    @property
    @abstractmethod
    def identified_speaker(self) -> "IIdentifiedSpeaker":
        pass

    @property
    @abstractmethod
    def new_speaker(self) -> IDiarizedSpeaker:
        pass


class IIdentifiedSpeaker(ISpeaker):
    @property
    @abstractmethod
    def speaker_id(self) -> SpeakerId:
        pass

    @abstractmethod
    def clone(self) -> "IIdentifiedSpeaker":
        pass

    @property
    @abstractmethod
    def distances(self) -> List[ISpeakersDistance]:
        pass

    @abstractmethod
    def add_distances(self, distances: ISpeakersDistance):
        pass


class IDiarizedSpeakers(ISerializable):
    @abstractmethod
    def append(self, value: IDiarizedSpeaker) -> None:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @property
    @abstractmethod
    def array(self) -> List[IDiarizedSpeaker]:
        pass


class IIdentifiedSpeakers(ISerializable):
    @abstractmethod
    def add(self, key: SpeakerId, value: IIdentifiedSpeaker) -> None:
        pass

    @property
    @abstractmethod
    def data(self) -> Dict[SpeakerId, IIdentifiedSpeaker]:
        pass

    @abstractmethod
    def __getitem__(self, key: SpeakerId) -> IIdentifiedSpeaker:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @property
    @abstractmethod
    def state_key(self) -> StateKey:
        pass

```

- /app/deno-sample/doll/engine/identifier/src/components/state_store/abstract.py
```python
from abc import ABC, abstractmethod
from components.env import IEnv
from components.abstracts import ISerializable, IState
from typing import Optional
from components.value_objects.state_key import StateKey


class IStateStore(ABC):
    @property
    @abstractmethod
    def env(self) -> IEnv:
        pass

    @abstractmethod
    def receive(self, key: StateKey) -> Optional[IState]:
        pass

    @abstractmethod
    def send(self, key: StateKey, value: IState) -> bool:
        pass

    @abstractmethod
    def remove(self, key: StateKey) -> bool:
        pass


# stateStoreは文脈的には以下の二つ存在する
# 1.diarizerが分離した音源を保持するためのstateStore
# 2.identifierが識別した話者の情報を一時的に保持するためのstateStore
# このため、stateStoreに接続するインスタンスは二つ存在することになる。
# diでinterfaceを介して接続する際に、それぞれが実際に何に接続するかは異なるかもしれない。
# 例えば、diarizerはs3、identifierはdynamodbに接続するかもしれない。それとも、両方ともredisに接続するかもしれない。
# このため、それぞれのstateStoreに対してinterfaceを継承したinterfaceを作成する。
class IDiarizerStateStore(IStateStore):
    pass


class IIdentifierStateStore(IStateStore):
    pass


if __name__ == "__main__":
    pass

    class a:
        def __init__(self, d: IDiarizerStateStore):
            self.d = d
            pass

```

- /app/deno-sample/doll/engine/identifier/src/components/state_store/lib_dynamo_db/__init__.py
```python

```

- /app/deno-sample/doll/engine/identifier/src/components/state_store/lib_dynamo_db/create_table.py
```python
from components.logger import Logger

logger = Logger.init(f"{__file__}:{__name__}")


def create_identifierd_speakers_table(client, table_name: str):
    """Check if the table exists, and create it if it does not."""
    try:
        client.meta.client.describe_table(TableName=table_name)

        logger.info(f"Table {table_name} already exists.")
    except client.meta.client.exceptions.ResourceNotFoundException:
        logger.info(f"Table {table_name} not found. Creating table...")
        client.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": "StateKey", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "StateKey", "AttributeType": "S"}
            ],
            ProvisionedThroughput={"ReadCapacityUnits": 10, "WriteCapacityUnits": 10},
        )
        logger.info(f"Table {table_name} created successfully.")

```

- /app/deno-sample/doll/engine/identifier/src/components/state_store/store_redis.py
```python
import redis
from injector import inject
from components.env import IEnv

from components.state_store.abstract import IStateStore
from components.abstracts import ISerializable, IState
import json
from components.value_objects.speaker_source import SpeakerSource
from typing import Dict
from components.value_objects.state import State
from typing import Optional
from components.value_objects.state_key import StateKey
from components.logger import Logger
# やがてこういう形で保持されることになる
# {
#     00: class IdentifiedSpeaker,
#     01: class IdentifiedSpeaker,
#     02: class IdentifiedSpeaker,
#     03: class IdentifiedSpeaker,
# }


class StoreRedis(IStateStore):
    @inject
    def __init__(self, env: IEnv):
        self._env = env
        self.client = redis.Redis(host=env.REDIS_HOST, port=6379)
        self.logger = Logger.init(f"{__file__}:{__name__}")

    @property
    def env(self) -> IEnv:
        return self._env

    def receive(
        self,
        key: StateKey,
    ) -> Optional[IState]:
        state_key = str(key)
        response = self.client.get(state_key)
        if not response:
            return None
        response_dict = json.loads(response)

        # TODO: ここに書くのは間違い。stateで管理するデータ型に問題がある
        key = list(response_dict.keys())[0]
        value = response_dict[key]

        state = State(key=str(key), value=value)
        return state

    # valueはserializedされたデータである程度しか保証できない
    def send(self, key: StateKey, value: IState) -> bool:
        seriarized_data = value.to_json()
        state_key = str(key)
        response = self.client.set(state_key, seriarized_data)

        if not response:
            raise Exception("Failed to set object")

        return True

    def remove(self, key: StateKey) -> bool:
        state_key = str(key)
        response = self.client.delete(state_key)

        if response == 0:
            return False  # キーが存在しないか、削除に失敗した場合
        return True


if __name__ == "__main__":
    from components.di import di
    from components.value_objects.speaker_id import SpeakerId
    from components.entity.identified_speaker import IdentifiedSpeaker
    from components.entity.diarized_speaker import DiarizedSpeaker
    import torch
    from components.value_objects.embedding_vector import EmbeddingVector
    import numpy as np
    from components.entity.identified_speakers import IdentifiedSpeakers
    from components.state_store.abstract import IIdentifierStateStore

    dic = di.inject()
    # compute_service: IComputeService = dic[IComputeService]()

    store: IStateStore = dic[IIdentifierStateStore]()

    state_key = StateKey("test1231234")
    sr = 16000

    # "例え空でもエラーにならない"
    try:
        data = store.receive(state_key)

        if not data:
            print("data is empty")
    except Exception as e:
        print(e)

    # init
    speaker_id = SpeakerId()
    speaker_source = SpeakerSource(value=torch.rand(1, 1000000), sample_rate=sr).ref()
    embedding_vector = EmbeddingVector(value=np.random.rand(1, 512).reshape(-1))
    speaker = DiarizedSpeaker(speaker_source=speaker_source)
    speaker.embedding_vector = embedding_vector
    identified_speaker = IdentifiedSpeaker(speaker, speaker_id)

    ispeakers = IdentifiedSpeakers(
        key=speaker_id, value=identified_speaker, state_key=state_key
    )

    for i in range(2):
        speaker_id = SpeakerId()
        speaker_source = SpeakerSource(
            value=torch.rand(1, 1000000), sample_rate=sr
        ).ref()

        embedding_vector = EmbeddingVector(value=np.random.rand(1, 512).reshape(-1))
        speaker = DiarizedSpeaker(speaker_source=speaker_source)
        speaker.embedding_vector = embedding_vector
        identified_speaker = IdentifiedSpeaker(speaker, speaker_id)
        ispeakers.add(speaker_id, identified_speaker)

    data = ispeakers.serialize()
    from components.utils.debug_tools import print_size

    size = print_size(data)

    print("size: ", size)

    # 値をsetする
    try:
        store.send(state_key, data)
    except Exception as e:
        print(e)

    # もう一回読み出す
    serializable_data: Optional[IState] = None
    try:
        serializable_data = store.receive(state_key)
    except Exception as e:
        print(e)

    if serializable_data is None:
        raise Exception("Failed to get object")

    if not isinstance(serializable_data, IState):
        raise Exception("Failed to get object. type is not IState")

    IdentifiedSpeakers.deserialize(serializable_data)

    # 削除
    try:
        store.remove(state_key)
    except Exception as e:
        print(e)

```

- /app/deno-sample/doll/engine/identifier/src/components/state_store/__init__.py
```python

```

- /app/deno-sample/doll/engine/identifier/src/components/state_store/store_s3.py
```python
import boto3
from components.state_store.abstract import IStateStore
from injector import inject
from components.env import IEnv
from components.value_objects.state_key import StateKey
from typing import Optional
from components.abstracts import IState
import uuid
from components.logger import Logger
from const import StateStoreType
from components.value_objects.s3_object_key import S3ObjectKey
from components.value_objects.bucket import BucketDestDiarizedAudio

from components.value_objects.state import State
from scipy.io import wavfile
import json


# TODO 実装がやばいのでリファクタする
class StoreS3(IStateStore):
    @inject
    def __init__(self, env: IEnv):
        # indentifierでは、dializerのdest bucketがsrc bucketになる
        # TODO: 名前を変更する
        self.logger = Logger.init(f"{__file__}:{__name__}")
        self._env = env

        # TODO: Dynamoと同じように定義する
        if env.DIARIZER_STATE_STORE_SERVICE == StateStoreType.S3_MINIO:
            self.client = boto3.client(
                "s3",
                region_name=env.AWS_REGION_NAME,
                endpoint_url=env.AWS_S3_ENDPOINT_URL,
                aws_access_key_id=env.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=env.AWS_SECRET_ACCESS_KEY,
            )
        elif env.DIARIZER_STATE_STORE_SERVICE == StateStoreType.S3:
            self.client = boto3.client("s3")
        else:
            raise ValueError(
                f"DIARIZER_STATE_STORE_SERVICE is invalid: {env.DIARIZER_STATE_STORE_SERVICE}"
            )

    @property
    def env(self) -> IEnv:
        return self._env

    # まず必要なのはreceiveなのか。

    def receive(self, state: IState) -> Optional[IState]:
        object_key = S3ObjectKey(state.value["objectKey"])
        bucket = BucketDestDiarizedAudio()

        # TODO: 可能であればバッファで処理するように変更
        tmp_file_path = f"/tmp/{uuid.uuid4()}"
        try:
            self.logger.debug(f"Downloading {object_key} from {bucket}")
            self.client.download_file(str(bucket), str(object_key), tmp_file_path)

        except Exception as e:
            self.logger.error(f"Failed to download {object_key} from {bucket}")
            raise e

        self.logger.debug(f"Downloaded {object_key} from {bucket}")

        with open(tmp_file_path, "rb") as f:
            serialized = f.read()
            state = State("DiarizedSpeaker", json.loads(serialized))

            return state

    # TODO: モックでしか使われない書き方になってるので、リファクタする
    def send(self, key: StateKey, value: IState) -> bool:
        # valueObjectに入れるか
        object_key = str(S3ObjectKey(key))
        dest_bucket = str(BucketDestDiarizedAudio())

        tmp_file_path = f"/tmp/{uuid.uuid4()}"
        serialized = value.serialize().to_json()

        with open(tmp_file_path, "w") as f:
            f.write(serialized)

        # TODO: 本当はon memoryで持つべき
        # diarizerが渡されるとして、その時位にはすでにseriarize されているので、ここでは投げるだけになる

        try:
            self.client.upload_file(tmp_file_path, dest_bucket, object_key)
        except Exception as e:
            raise e

        pass

    def remove(self, key: StateKey) -> bool:
        raise NotImplementedError
        pass


if __name__ == "__main__":
    from components.di import di
    from components.state_store.abstract import IDiarizerStateStore

    dic = di.inject()
    # TODO
    # get_state_store_service S3_MINIO
    # get_state_store_service REDIS
    # get_state_store_service S3_MINIO
    # get_state_store_service REDIS

    # 2回しか叩かれない想定なのに4回叩かれてるのはなぜ

    state_store: IDiarizerStateStore = dic[IDiarizerStateStore]()
    print(state_store)

```

- /app/deno-sample/doll/engine/identifier/src/components/state_store/store_dynamo.py
```python
import boto3
from injector import inject
from components.env import IEnv
from components.state_store.abstract import IStateStore
from components.abstracts import IState
from components.value_objects.state import State
from typing import Optional
from components.value_objects.state_key import StateKey
from components.state_store.lib_dynamo_db.create_table import (
    create_identifierd_speakers_table,
)


class StoreDynamo(IStateStore):
    @inject
    def __init__(self, env: IEnv):
        self._env = env
        # TODO: DIで渡す
        # TODO: DEVはconstで定義する
        if env.IDENTIFIER_ENV == "DEV":
            self.client = boto3.resource(
                "dynamodb",
                region_name=env.AWS_REGION_NAME,
                endpoint_url=env.AWS_DYNAMODB_ENDPOINT_URL,
                aws_access_key_id=env.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=env.AWS_SECRET_ACCESS_KEY,
            )
        else:
            self.client = boto3.resource("dynamodb")

        # ここじゃない方がいいと思う。
        # これだとテーブル増えた時がしんどい。
        # 引数で外部からテーブル名を指定した方がいいんじゃないか
        # TODO s3のバケットみたいにvalu objectsにしてもいいんじゃない??
        self.table_name = "publicapi-chk-doll-results"
        # ここでテーブルなかったら作成するようにしてほしい
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        create_identifierd_speakers_table(self.client, self.table_name)

    @property
    def env(self) -> IEnv:
        return self._env

    def receive(self, key: StateKey) -> Optional[IState]:
        state_key = str(key)
        table = self.client.Table(self.table_name)
        response = table.get_item(Key={"StateKey": state_key})
        item = response.get("Item")
        if not item:
            return None
        value = item.get("Value")

        key = list(value.keys())[0]
        state = State(key=str(key), value=value)
        return state

    def send(self, key: StateKey, value: IState) -> bool:
        state_key = str(key)
        table = self.client.Table(self.table_name)
        serialized_data = value.to_dict()

        table.put_item(
            Item={
                "StateKey": state_key,
                "Value": serialized_data,
            }
        )

        return True

    def remove(self, key: StateKey) -> bool:
        state_key = str(key)
        table = self.client.Table(self.table_name)
        response = table.delete_item(Key={"StateKey": state_key})

        return response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 200


if __name__ == "__main__":
    from components.env import Env
    from components.value_objects.speaker_id import SpeakerId
    from components.entity.identified_speaker import IdentifiedSpeaker
    from components.entity.identified_speakers import IdentifiedSpeakers
    from components.entity.diarized_speaker import DiarizedSpeaker
    from components.value_objects.state import State
    from components.utils import get_file_path, load_audio
    from components.value_objects.speaker_source import SpeakerSource
    import numpy as np
    from components.value_objects.embedding_vector import EmbeddingVector

    section_dir = "/app/sample_audio/section1"
    # dirからfileのパスを取得する
    file_paths = get_file_path(section_dir)

    # torchAudioで読み込む
    section_audio_data = [load_audio(file_path=file_path) for file_path in file_paths]

    new_speakers = [
        DiarizedSpeaker(SpeakerSource(value=waveform, sample_rate=sr).ref())
        for waveform, sr in section_audio_data
    ]

    for n_speaker in new_speakers:
        # 形状 (512,) の ndarray を作成
        embedding_data = np.random.rand(512)
        embedding_vector = EmbeddingVector(value=embedding_data)
        n_speaker.embedding_vector = embedding_vector

    # 環境変数を設定したEnvインスタンスの作成
    env = Env()

    # StoreDynamoのインスタンスを作成
    store = StoreDynamo(env)

    # テスト用のキーとデータを準備
    test_key = StateKey("test_key432425235232")
    test_speaker_id = SpeakerId()
    first_speaker = new_speakers[0]
    test_speaker = IdentifiedSpeaker(
        first_speaker=first_speaker, speaker_id=test_speaker_id
    )  # ダミーのIdentifiedSpeaker
    identified_speakers = IdentifiedSpeakers(
        key=test_speaker_id, value=test_speaker, state_key=test_key
    )

    # データの送信
    print("Sending data to DynamoDB...")
    store.send(test_key, identified_speakers.serialize())

    # データの受信
    print("Receiving data from DynamoDB...")
    received_data = store.receive(test_key)
    print(f"Received data: {type(received_data)}")

    # データの削除
    print("Removing data from DynamoDB...")
    store.remove(test_key)
    print("Data removed.")

```

- /app/deno-sample/doll/engine/identifier/src/service/compute_service.py
```python
import const
from service.abstract_service import IComputeService
from injector import inject
from components.speaker_identification import abstract
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.distance import cdist
from components.value_objects.speakers_distance import SpeakersDistance
from const.speaker import SpeakerDataDict
from components.value_objects.speaker_id import SpeakerId
from components.abstracts import (
    ISpeaker,
    IDiarizedSpeaker,
    IIdentifiedSpeakers,
)
from components.entity.identified_speaker import IdentifiedSpeaker
from components.value_objects.sorted_array import AscendingSortedArray
from components.logger import Logger


class ComputeService(IComputeService):
    @inject
    def __init__(self, compute: abstract.ICompute):
        self.compute = compute
        pass
        self.logger = Logger.init(f"{__file__}:{__name__}")

    def compute_speakers_embedding_parallel(
        self, speakers: List[IDiarizedSpeaker]
    ) -> List[IDiarizedSpeaker]:
        """
        与えられた分離済み話者のembeddingを並列で計算するサービス
        """

        result: List[IDiarizedSpeaker] = []

        # 並列処理でembeddingを計算する
        with ThreadPoolExecutor() as executor:
            futures = []
            for speaker in speakers:
                futures.append(executor.submit(self.compute.compute_distance, speaker))
            for future in as_completed(futures):
                try:
                    speaker = future.result()
                    result.append(speaker)
                except Exception as e:
                    raise e

        return result

    def compute_speakers_distance(
        self, speakers: SpeakerDataDict
    ) -> AscendingSortedArray[SpeakersDistance]:
        # 結果を格納するリスト
        results: List[SpeakersDistance] = []

        new_speakers: List[IDiarizedSpeaker] = speakers["new_speakers"]
        identified_speakers: IIdentifiedSpeakers = speakers["identified_speakers"]

        # 話者識別済みと新規話者の距離を求める

        # 距離の計算はここで行う??
        def inner(speaker1: ISpeaker, speaker2: ISpeaker) -> SpeakersDistance:
            cosine_d: float = cdist(
                [speaker1.embedding_vector.value],
                [speaker2.embedding_vector.value],
                metric="cosine",
            )[0][0]
            distance = SpeakersDistance(cosine_d, speaker1, speaker2)
            return distance

        self.logger.debug("距離を計算します")
        self.logger.debug(f"識別ずみ話者数: {len(identified_speakers)}")
        self.logger.debug(f"新規話者数: {len(new_speakers)}")
        results = [
            inner(i_speaker, n_speaker)
            for _, i_speaker in identified_speakers.data.items()
            for n_speaker in new_speakers
        ]
        self.logger.debug("距離計算完了")
        self.logger.debug(f"距離の数: {len(results)}")

        results: AscendingSortedArray[SpeakersDistance] = AscendingSortedArray(results)

        return results

    def _clustering_by_limit(
        self,
        identified_speakers: IIdentifiedSpeakers,
        distances: AscendingSortedArray[SpeakersDistance],
    ) -> Tuple[IIdentifiedSpeakers, List[SpeakersDistance]]:
        processed_task: List[IDiarizedSpeaker] = []
        long_distances: List[SpeakersDistance] = []

        for i, distance in enumerate(distances):
            speaker_id = distance.identified_speaker.speaker_id
            i_speaker = identified_speakers[speaker_id]

            # すでに他の話者に振り分けられているか確認する
            if distance.new_speaker in processed_task:
                self.logger.debug(
                    f"話者{distance.new_speaker}はすでに {i_speaker}に振り分けられています"
                )
                continue

            if distance < const.SIMILARITY_THRESHOLD:
                # 閾値以下の場合、距離が近いと判断して同一スピーカーとして扱う

                self.logger.debug(
                    f"距離が{float(distance)}だったので{distance.new_speaker}を{i_speaker}に振り分けます"
                )
                i_speaker.add_distances(distance)
                processed_task.append(distance.new_speaker)
            else:
                long_distances.append(distance)

        self.logger.debug(f"振り分けられなかった話者数: {len(long_distances)}")
        return identified_speakers, long_distances

    def _add_identified_speaker(
        self,
        identified_speakers: IIdentifiedSpeakers,
        distances: List[SpeakersDistance],
    ) -> Tuple[IIdentifiedSpeakers, List[SpeakersDistance]]:
        # 振り分けられなかった話者は新規話者として扱う
        for distance in distances:
            # 正直なんでもいいのでuuidで笑
            speaker_id = SpeakerId()
            d_speaker = distance.new_speaker

            identified_speaker = IdentifiedSpeaker(
                speaker_id=speaker_id, first_speaker=d_speaker
            )
            identified_speakers.add(key=speaker_id, value=identified_speaker)

        return identified_speakers, distances

    def clustering(
        self,
        identified_speakers: IIdentifiedSpeakers,
        distances: AscendingSortedArray[SpeakersDistance],
    ) -> IIdentifiedSpeakers:
        # 生成したベクトル表現を閾値に従って分類する
        identified_speakers, distances = self._clustering_by_limit(
            identified_speakers, distances
        )

        # ここで振り分けられなかったものは新規話者として扱う
        identified_speakers, distances = self._add_identified_speaker(
            identified_speakers, distances
        )

        # TODO: Embeddingに計算元のデータのサイズを渡しておく。
        # もしより大きなデータサイズのものが渡された場合、今後はそちらを優先して使うようにする
        # それか平均とるかよなあ

        return identified_speakers

```

- /app/deno-sample/doll/engine/identifier/src/service/__init__.py
```python

```

- /app/deno-sample/doll/engine/identifier/src/service/abstract_service.py
```python
from abc import ABC, abstractmethod
from typing import List
from components.value_objects.speakers_distance import SpeakersDistance
from const.speaker import SpeakerDataDict
from components.abstracts import IDiarizedSpeaker, IIdentifiedSpeakers
from components.value_objects.sorted_array import AscendingSortedArray


class IComputeService(ABC):
    @abstractmethod
    def compute_speakers_distance(
        self, speakers: SpeakerDataDict
    ) -> AscendingSortedArray[SpeakersDistance]:
        pass

    @abstractmethod
    def clustering(
        self,
        identified_speakers: IIdentifiedSpeakers,
        distances: AscendingSortedArray[SpeakersDistance],
    ) -> IIdentifiedSpeakers:
        pass

    @abstractmethod
    def compute_speakers_embedding_parallel(
        self, speakers: List[IDiarizedSpeaker]
    ) -> List[IDiarizedSpeaker]:
        pass


class IStateStoreService(ABC):
    @abstractmethod
    def receive(self, key: str):
        pass

    @abstractmethod
    def send(self, key: str, value):
        pass

```

- /app/deno-sample/doll/engine/identifier/src/usecase/__init__.py
```python

```

- /app/deno-sample/doll/engine/identifier/src/usecase/abstract_usecase.py
```python
from abc import ABC, abstractmethod
from components.abstracts import IDiarizedSpeaker, IDiarizedSpeakers
from typing import List, Optional, Dict
from const.speaker import SpeakerDataDict
from components.abstracts import IIdentifiedSpeakers
from components.value_objects.state_key import StateKey


class ISpeakerIdentiicationUsecase(ABC):
    @abstractmethod
    # TODO これダメ。。。vo渡すようにする
    def pull_dialized_speaker(
        self, dializer_event: Dict
    ) -> Optional[IDiarizedSpeakers]:
        pass

    @abstractmethod
    def pull_state(
        self,
        new_speakers: IDiarizedSpeakers,
        identifier_key: StateKey,
    ) -> Optional[SpeakerDataDict]:
        pass

    @abstractmethod
    def identify(self, speakers: SpeakerDataDict) -> IIdentifiedSpeakers:
        pass

    @abstractmethod
    def push_state(
        self, identifier_key: StateKey, identified_speakers: IIdentifiedSpeakers
    ) -> bool:
        pass

```

- /app/deno-sample/doll/engine/identifier/src/usecase/speaker_identification_usecase.py
```python
from usecase.abstract_usecase import ISpeakerIdentiicationUsecase
from service.abstract_service import IComputeService
from typing import List, Optional, Dict
from injector import inject
from components.state_store.abstract import (
    IDiarizerStateStore,
    IIdentifierStateStore,
)
from components.entity.identified_speaker import IdentifiedSpeaker
from const.speaker import SpeakerDataDict
from components.entity.identified_speakers import IdentifiedSpeakers
from components.value_objects.speaker_id import SpeakerId
from components.value_objects.state_key import StateKey
from components.abstracts import (
    IDiarizedSpeaker,
    IIdentifiedSpeakers,
    IDiarizedSpeakers,
)
from components.entity.diarized_speakers import DiarizedSpeakers
from components.value_objects.state import State
from components.entity.diarized_speaker import DiarizedSpeaker
from components.logger import Logger


class SpeakerIdentificationUsecase(ISpeakerIdentiicationUsecase):
    @inject
    def __init__(
        self,
        compute_embedding_service: IComputeService,
        dializer_state_store: IDiarizerStateStore,
        identifier_state_store: IIdentifierStateStore,
    ):
        self.compute_embedding_service = compute_embedding_service
        self.dializer_state_store = dializer_state_store
        self.identifier_state_store = identifier_state_store

        self.logger = Logger.init(f"{__file__}:{__name__}")

    # TODO ひどいので直す
    def pull_dialized_speaker(
        self, dializer_event: Dict
    ) -> Optional[IDiarizedSpeakers]:
        dializer_event["ObjectKeys"] = [
            StateKey(key) for key in dializer_event["ObjectKeys"]
        ]

        d_speakers: List[IDiarizedSpeaker] = []
        for k, v in dializer_event.items():
            if k == "ObjectKeys":
                for key in v:
                    state = State(
                        key=k,
                        value={
                            "CompanyUUID": dializer_event["CompanyUUID"],
                            "VisitID": dializer_event["VisitID"],
                            "Destination": dializer_event["Destination"],
                            "objectKey": key,
                        },
                    )
                    d_state = self.dializer_state_store.receive(state)

                    if not d_state:
                        return None

                    d_speaker = DiarizedSpeaker.deserialize(d_state)
                    d_speakers.append(d_speaker)

        self.logger.debug(f"dialized_speakers size: {len(d_speakers)}")
        first_speaker = d_speakers.pop()
        dialized_speakers = DiarizedSpeakers(first_speaker)

        for speaker in d_speakers:
            dialized_speakers.append(speaker)

        return dialized_speakers

    def pull_state(
        self,
        new_speakers: IDiarizedSpeakers,
        identifier_key: StateKey,
    ) -> Optional[SpeakerDataDict]:
        """
        同期処理を行います。
        1. 引数から新規で追加されたスピーカーを取得します。
        2. storeに保存されているスピーカーを取得します。
        3. storeに保存されているスピーカーがなければ、新規スピーカーにラベルを割り当ててそのままstoreに保存します。
        """
        # 開発のデフォルトはredis
        # keyを使ってstoreからデータを取得する
        state = self.identifier_state_store.receive(identifier_key)

        new_speakers = (
            self.compute_embedding_service.compute_speakers_embedding_parallel(
                new_speakers
            )
        )

        # storeにデータがないなら、新規スピーカーにラベルを割り当ててそのままstoreに保存する
        if not state:
            first_speaker = new_speakers[0]
            speaker_id = SpeakerId()
            ispeakers = IdentifiedSpeakers(
                key=speaker_id,
                value=IdentifiedSpeaker(first_speaker, speaker_id),
                state_key=identifier_key,
            )

            for speaker in new_speakers[1:]:
                speaker_id = SpeakerId()

                identified_speaker = IdentifiedSpeaker(speaker, speaker_id)
                ispeakers.add(key=speaker_id, value=identified_speaker)
            state = ispeakers.serialize()

            self.identifier_state_store.send(identifier_key, state)
            return None

        # storeにデータがあるなら、識別処理に進む
        ispeakers = IdentifiedSpeakers.deserialize(state)

        speakers = SpeakerDataDict(
            identified_speakers=ispeakers, new_speakers=new_speakers
        )
        return speakers

    def identify(self, speakers: SpeakerDataDict) -> IIdentifiedSpeakers:
        # 話者間の距離を計算する
        self.logger.info("compute_speakers_distance start")
        self.logger.info(f"話者間の距離を計算します")
        distances = self.compute_embedding_service.compute_speakers_distance(
            speakers=speakers
        )

        for d in distances:
            self.logger.info(
                f"{d.identified_speaker}と{d.new_speaker}の距離は{float(d)}"
            )

        identified_speakers = speakers["identified_speakers"]
        self.logger.info("clustering start")
        identified_speakers = self.compute_embedding_service.clustering(
            identified_speakers=identified_speakers, distances=distances
        )

        return identified_speakers

    def push_state(
        self, identifier_key: StateKey, identified_speakers: IIdentifiedSpeakers
    ) -> bool:
        """
        現在の話者情報を保存します。
        """
        state = identified_speakers.serialize()

        self.identifier_state_store.send(identifier_key, state)

        return True

```

- /app/deno-sample/doll/engine/dializer/tests/speaker_dialization/test_speaker_dialization.py
```python
import pytest
from speaker_dialization import SpeakerDialization, SPEAKER_LABEL
from speaker_dialization.speaker import Speaker
from lib import create_mock_speaker
from typing import Dict
from components.const import MIN_SPEECH_DURATION_SEC


@pytest.fixture
def setup_module():
    sd = SpeakerDialization()
    # sd.exec()したときに副作用が発生するからここでは入れない。
    # 副作用起きちゃダメなんだけどね。。。
    speakers = create_mock_speaker()

    return {"instance": sd, "dialized_speakers": speakers}


def test_prune_short_audio(setup_module):
    # Warningいっぱい出るけど一旦無視でいいです。
    # TODO: warningについて調べる
    sd: SpeakerDialization = setup_module["instance"]
    speakers: Dict[SPEAKER_LABEL, Speaker] = setup_module["dialized_speakers"]

    sd.derized_speakers = speakers

    sd.prune_short_audio(MIN_SPEECH_DURATION_SEC)

    assert len(sd.derized_speakers) == 3

```

- /app/deno-sample/doll/engine/dializer/tests/speaker_dialization/lib.py
```python
from speaker_dialization.speaker import Speaker
import torch
from typing import List, Dict
from speaker_dialization import SPEAKER_LABEL


def create_mock_speaker() -> Dict[SPEAKER_LABEL, Speaker]:
    """
    speaker_dialization/speaker.pyのSpeakerクラスのモックを作成する
    """
    speaers: Dict[SPEAKER_LABEL, Speaker] = {}
    # サンプルレートの設定
    sample_rate = 16000  # 16kHz

    # 8秒以上の波形データを3個作成
    waveform_1 = torch.randn((1, 8 * sample_rate))  # 8秒
    waveform_2 = torch.randn((1, 10 * sample_rate))  # 10秒
    waveform_3 = torch.randn((1, 12 * sample_rate))  # 12秒

    # 8秒未満の波形データを2個作成
    waveform_4 = torch.randn((1, 4 * sample_rate))  # 4秒
    waveform_5 = torch.randn((1, 6 * sample_rate))  # 6秒
    # テスト用の話者データを作成

    test_speakers = [
        Speaker(
            start_speaking_time=0,
            end_speaking_time=8,
            speaker_waveform=waveform_1,
            sample_rate=sample_rate,
            speaker="00",
            audio_id="audio_001",
            file_name="test_audio_1.wav",
        ),
        Speaker(
            start_speaking_time=0,
            end_speaking_time=10,
            speaker_waveform=waveform_2,
            sample_rate=sample_rate,
            speaker="01",
            audio_id="audio_002",
            file_name="test_audio_2.wav",
        ),
        Speaker(
            start_speaking_time=0,
            end_speaking_time=12,
            speaker_waveform=waveform_3,
            sample_rate=sample_rate,
            speaker="02",
            audio_id="audio_003",
            file_name="test_audio_3.wav",
        ),
        Speaker(
            start_speaking_time=0,
            end_speaking_time=4,
            speaker_waveform=waveform_4,
            sample_rate=sample_rate,
            speaker="03",
            audio_id="audio_004",
            file_name="test_audio_4.wav",
        ),
        Speaker(
            start_speaking_time=0,
            end_speaking_time=6,
            speaker_waveform=waveform_5,
            sample_rate=sample_rate,
            speaker="04",
            audio_id="audio_005",
            file_name="test_audio_5.wav",
        ),
    ]

    speaers = {speaker.speaker: speaker for speaker in test_speakers}

    return speaers

```

- /app/deno-sample/doll/engine/dializer/lambda_function.py
```python
import os
# pytonでwhooami
print("whoami: ", os.system("whoami"))

cwd = os.getcwd()
print("cwd: ", cwd)
import sys

sys.path.append(os.path.join(cwd, "src"))
sys.path.append(os.path.join(cwd, ".venv", "lib", "python3.12", "site-packages"))
from src.app.app import handler as app_handler


def handler(event, context):
    return app_handler(event, context)

```

- /app/deno-sample/doll/engine/dializer/tool/mock_subscriber.py
```python
import redis
import myredis


def subscriber():
    client = redis.Redis(host=myredis.HOST, port=6379, db=0)
    pubsub = client.pubsub()

    channnel_name = myredis.SD_SUCCESSED_CHANNEL
    pubsub.subscribe(channnel_name)

    print(f"Subscribed to {channnel_name}")

    received_data = []

    for message in pubsub.listen():
        if message["type"] == "message":
            received_data.append(message["data"].decode("utf-8"))

            print("received_data current count", received_data)


if __name__ == "__main__":
    subscriber()

```

- /app/deno-sample/doll/engine/dializer/tool/create_test_data.py
```python
import redis
from pydub import AudioSegment

import audio

HOST = "doll-cli-redis"
AUDIO_CHANNEL = "audio-channel"
# 話者分離成功を通知して後続タスクに引き継ぐためのチャンネル
SD_SUCCESSED_CHANNEL = "sd-successed-channel"

client = redis.Redis(host=HOST, port=6379)

# audio_path = "/app/sample_audio/chunk0_short.wav"
# audio_path = "/app/sample_audio/0a0f96f11ecae60ec2872db54a78a58a3522e52bc873d0e1aa45c3a22ef1b33f_SPEAKER_00_segment_1486000-289846000.wav"
audio_path = "/app/sample_audio/chunk0.wav"

# Load audio file
audio_data = AudioSegment.from_file(audio_path)

serialize_raw_audio = audio.B64EncodedRawAudio(
    audio_data, 3 * 1000, "test"
).to_serialize()

key = "test_0"

response = client.set(key, serialize_raw_audio)

print(response)

```

- /app/deno-sample/doll/engine/dializer/tool/create_test_bucket_data.py
```python
import boto3
from pydub import AudioSegment

import audio
from components.env import Env
from components.value_objects.bucket import (
    BucketDestDiarizedAudio,
    BucketSrcReceivedRawAudio,
)


env = Env()

AWS_ACCESS_KEY_ID = env.AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY = env.AWS_SECRET_ACCESS_KEY
AWS_ENDPOINT_URL = env.AWS_S3_ENDPOINT_URL
print(AWS_ENDPOINT_URL)
AWS_REGION = env.AWS_REGION_NAME

AUDIO_PATH = "/app/sample_audio/chunk0.wav"
AUDIO_B64_PATH = "/app/sample_audio/chunk0_b64.wav"
SRC_BUCKET = str(BucketSrcReceivedRawAudio())
DEST_BUCKET = str(BucketDestDiarizedAudio())
UPLOAD_OBJECT_KEY = "companies/4ce9cf31-4bae-4cf5-bc18-55ddb061d024/visits/1/1720742400_1720746000.wav.gz"


audio_data = AudioSegment.from_file(AUDIO_PATH)
# TODO: serializeする時にcompressuしてることが明示的にわかった方がいい
serialize_raw_audio = audio.B64EncodedRawAudio(
    audio_data, 45000, "test"
).to_serialize()

with open(AUDIO_B64_PATH, "wb") as f:
    f.write(serialize_raw_audio)

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    endpoint_url=AWS_ENDPOINT_URL,
)

try:
    s3.create_bucket(Bucket=SRC_BUCKET)
except Exception as e:
    print(e)

try:
    s3.create_bucket(Bucket=DEST_BUCKET)
except Exception as e:
    print(e)

s3.upload_file(AUDIO_B64_PATH, SRC_BUCKET, UPLOAD_OBJECT_KEY)

print("create test bucket data successed")

```

- /app/deno-sample/doll/engine/dializer/src/handler/lib.py
```python
"""
ライブラリとして呼ばれるときを想定
"""
```

- /app/deno-sample/doll/engine/dializer/src/handler/__init__.py
```python
def hello() -> str:
    return "Hello from app!"

```

- /app/deno-sample/doll/engine/dializer/src/handler/lambda_handler.py
```python
"""
lambda実行相当(ローカル含む)で呼ばれることを期待しています。
"""

import sys
sys.path.append("/app/src")
sys.path.append("/app/src/app")

from dializer import dializer


# input_eventにこういうのが飛んでくる想定
# {
# 	    "CompanyUUID": "4ce9cf31-4bae-4cf5-bc18-55ddb061d024",
# 	    "ObjectKey": "companies/4ce9cf31-4bae-4cf5-bc18-55ddb061d024/visits/1/1720742400_1720746000.wav.b64",
# 	    "VisitID": "1234",
# 	    "IsB64": "True",
# }
def lambda_handler(input_event, context):
    return dializer(input_event, context)
```

- /app/deno-sample/doll/engine/dializer/src/handler/inject.py
```python
import data_transfer.redis_transfer
import data_transfer.transfer
import injector
from typing import Callable, TypeVar

import data_transfer

T = TypeVar("T")


class DependencyContainer:
    def __init__(self, use_redis: bool):
        self._use_redis = use_redis
        self._injector = injector.Injector(self.configure)

    def configure(self, binder: injector.Binder) -> None:
        if self._use_redis:
            binder.bind(
                data_transfer.ITransfer, to=data_transfer.redis_transfer.RedisTransfer
            )
        else:
            binder.bind(data_transfer.ITransfer, to=data_transfer.transfer.Transfer)

    def __getitem__(self, klass: T) -> Callable:
        # ex.) DependencyContainer[ITransfer]() のように実体クラスを取得する
        return lambda: self._injector.get(klass)

```

- /app/deno-sample/doll/engine/dializer/src/handler/app.py
```python
import json
import sys
import time

sys.path.append("/app/src")
sys.path.append("/app/src/app")
import audio
import audio.raw_audio
import data_transfer
import speaker_dialization

from components import const
from components.logger import Logger
import json
from components.value_objects.state import State
from components.di import di
from components.value_objects.state_key import StateKey
from components.value_objects.object_key import ObjectKey
from components.value_objects.bucket import BucketSrcReceivedRawAudio
from data_transfer import ReceiveData
# 投げられた音声ファイルを受け取り、推論を行う


# DIコンテナの取得
dic = di.inject()
logger = Logger.init(f"{__file__}:{__name__}")


# input_eventにこういうのが飛んでくる想定
# {
# 	    "CompanyUUID": "4ce9cf31-4bae-4cf5-bc18-55ddb061d024",
# 	    "ObjectKey": "companies/4ce9cf31-4bae-4cf5-bc18-55ddb061d024/visits/1/1720742400_1720746000.wav.b64",
# 	    "VisitID": "1234",
# 	    "IsB64": "True",
# }
def handler(input_event, context):
    # typeがわからない、多分辞書だと思うけど
    logger.debug(f"event type: {type(input_event)}")
    if input_event is None:
        raise ValueError("event is required")
    if isinstance(input_event, dict):
        logger.debug(f"event: {json.dumps(input_event)}, context: {type(context)}")
    else:
        logger.warning(f"event: {json.dumps(input_event)}, context: {type(context)}")
        raise TypeError("event must be a dict")

    # 受け取ったinputはstateとして保持する
    # TODO presentation層を作ってそっちで行う

    if "CompanyUUID" not in input_event:
        raise ValueError("CompanyUUID is required")
    if "VisitID" not in input_event:
        raise ValueError("VisitID is required")
    if "IsB64" not in input_event:
        raise ValueError("IsB64 is required")

    state = State(
        "input_event",
        {
            "CompanyUUID": input_event["CompanyUUID"],
            "ObjectKey": ObjectKey(input_event),
            "VisitID": StateKey(input_event["VisitID"]),
            "IsB64": input_event["IsB64"],
            "Bucket": BucketSrcReceivedRawAudio(),
        },
    )

    # transferは外部ストレージとのデータ送受信を行う
    transfer: data_transfer.ITransfer = dic[data_transfer.ITransfer]()
    logger.debug(f"transfer: {transfer}")

    # eventを元にデータ受信
    logger.debug(f"音声データ(生)受信します")
    receive_data = transfer.receive(state)
    logger.debug(f"生音声データ受信完了したよ")
    if receive_data is None:
        raise ValueError("receive_data is required")

    receive_data = ReceiveData(receive_data)

    # 音声データをデシリアライズ
    logger.info(f"音声データデシリアライズします")
    b64_raw_audio = audio.B64EncodedRawAudio.from_serialize(receive_data.raw_audio)
    logger.info(f"音声データデシリアライズ完了しました")

    # テンソルに変換
    raw_audio = audio.raw_audio.RawAudio(b64_raw_audio)

    # 話者分離
    start = time.perf_counter()
    logger.info(f"話者分離を行います")
    sd = speaker_dialization.SpeakerDialization()
    sd.exec(raw_audio)
    sd.prune_short_audio(const.MIN_SPEECH_DURATION_SEC)
    logger.info(f"話者分離完了しました。 exec time: {time.perf_counter() - start} sec")

    # scaleして外部ストレージに保存
    # put_keys は、putしたキー情報とメタデータをまとめてoutputとするために使用する
    put_keys = []
    for speaker in sd.diarized_speakers:
        try:
            serialized = speaker.serialize().to_json()
            logger.info(f"音源分離結果を外部ストレージに保存します")
            put_key = transfer.send(
                receive_data.dest_key_prefix,
                b64_raw_audio.audio_id,
                receive_data.section_id,
                str(speaker),
                serialized,
            )
            logger.info(
                f"音源分離結果を外部ストレージに保存しました。put_key: {put_key}"
            )
            put_keys.append(put_key)
        except Exception as e:
            raise e

    identifier_input = {
        "CompanyUUID": receive_data.company_uuid,
        "VisitID": str(receive_data.visit_id),
        "Destination": transfer.get_destination(),
        "ObjectKeys": put_keys,
    }

    logger.debug(f"identifier_input: {json.dumps(identifier_input)}")

    logger.info(f"successed to diarizer process")

    return identifier_input


# script実行起点
def script():
    args = sys.argv
    event = args[1]
    event = json.loads(event)
    print("event", event)

    handler(event, None)


# if __name__ == "__main__":
#     args = sys.argv
#     event = args[1]
#     print("event", event)

#     handler(None, None)

# {
#         "FilePath": "companies/4ce9cf31-4bae-4cf5-bc18-55ddb061d024/visits/1/1720742400_1720746000.wav.b64",
#         "VisitID": 1234
#     }

```

- /app/deno-sample/doll/engine/dializer/src/handler/script.py
```python
"""
ローカルで開発時に呼ばれることを想定しています。
"""
import sys
import json
from dializer import dializer


# script実行起点
def script():
    args = sys.argv
    event = args[1]
    event = json.loads(event)
    print("event", event)

    dializer(event, None)
```

- /app/deno-sample/doll/engine/dializer/src/data_transfer/__init__.py
```python
import abc
import uuid

from audio import SerializeRawAudio
from components.abstracts import IState
from typing import Optional
from components.value_objects.object_key import ObjectKey
from components.value_objects.bucket import BucketSrcReceivedRawAudio
from components.value_objects.state_key import StateKey


# TODO: company_uuid, visit_id をvoにしたい(KeyPrefixというvoでもいいかも)
class ReceiveData:
    def __init__(self, state: IState):
        value = state.value
        self.raw_audio: SerializeRawAudio = value["SerializeRawAudio"]
        self.section_id: str = value["SectionId"]
        # dest_key_prefix は、送信先のキーに用いられるprefixです
        self.dest_key_prefix: str = value["DestKeyPrefix"]
        self.company_uuid: str = value["CompanyUUID"]
        self.visit_id: StateKey = value["VisitID"]
        self.is_b64: bool = value["IsB64"]
        self.bucket: BucketSrcReceivedRawAudio = value["Bucket"]
        self.object_key: ObjectKey = value["ObjectKey"]


class ITransfer(metaclass=abc.ABCMeta):
    """主にローカルと実環境の切り替えのためのデータ送受信を担うインターフェース"""

    @abc.abstractmethod
    def receive(self, state: IState) -> Optional[IState]:
        raise NotImplementedError()

    @abc.abstractmethod
    def send(
        self,
        key_prefix: str,
        audio_id: str,
        section_id: str,
        speaker_label: str,
        serialized: bytes,
    ) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_destination(self) -> str:
        raise NotImplementedError()

```

- /app/deno-sample/doll/engine/dializer/src/data_transfer/transfer.py
```python
import base64
import boto3
import gzip
import json
import numpy as np
from scipy.io import wavfile
import uuid
from injector import inject
from data_transfer import ITransfer, ReceiveData
from audio import SerializeRawAudio
from components.env import IEnv
from components.const import StateStoreType
from components.abstracts import IState
from components.logger import Logger
from components.value_objects.object_key import ObjectKey
from components.value_objects.bucket import (
    BucketSrcReceivedRawAudio,
    BucketDestDiarizedAudio,
)
from typing import Optional
from components.value_objects.state import State


class Transfer(ITransfer):
    """S3をデータの送受信先として用いるクラス"""

    @inject
    def __init__(self, e: IEnv):
        self.dest_bucket: BucketDestDiarizedAudio = BucketDestDiarizedAudio()
        self.logger = Logger.init(f"{__file__}:{__name__}")

        if e.DIARIZER_STATE_STORE_SERVICE == StateStoreType.S3_MINIO:
            self.client = boto3.client(
                "s3",
                region_name=e.AWS_REGION_NAME,
                aws_access_key_id=e.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=e.AWS_SECRET_ACCESS_KEY,
                endpoint_url=e.AWS_S3_ENDPOINT_URL,
            )
        elif e.DIARIZER_STATE_STORE_SERVICE == StateStoreType.S3:
            self.client = boto3.client("s3")
        else:
            raise ValueError(
                f"DIARIZER_STATE_STORE_SERVICE is invalid: {e.DIARIZER_STATE_STORE_SERVICE}"
            )

    def receive(self, state: IState) -> Optional[IState]:
        """
        step_functionsのinputが発生します
        飛んでくるメッセージがこれ(参考)
        https://github.com/acall-inc/infra-aws-sub-stacks/pull/219/files#diff-9e551d7609609d1c595312522028c9b103d34ff060151d39b77f8ba2fa2dc1e9R4
        # 参考
        https://medium.com/@yuvarajmailme/step-function-activity-with-python-c007178037af
        https://docs.aws.amazon.com/step-functions/latest/dg/tutorial-creating-lambda-state-machine.html#create-lambda-function
        """

        object_key: ObjectKey = str(state.value["ObjectKey"])
        bucket: BucketSrcReceivedRawAudio = str(state.value["Bucket"])

        # TODO: 可能であればバッファで処理するように変更
        tmp_file_path = f"/tmp/{uuid.uuid4()}"
        try:
            self.logger.debug(f"Downloading {object_key} from {bucket}")
            self.client.download_file(str(bucket), str(object_key), tmp_file_path)

        except Exception as e:
            self.logger.error(f"Failed to download {object_key} from {bucket}")
            raise e
        self.logger.debug(f"Downloaded {object_key} from {bucket}")

        with open(tmp_file_path, "rb") as f:
            serialized = f.read()
            # TODO: section_idは暫定的にファイル名を拡張子を除いた用いているが、適宜修正
            # TODO section_idはいらないかも、dest_key_prefixは？？？
            section_id = object_key.split("/")[-1].split(".")[0]
            # 送信先のキーを受信時のオブジェクトキーのファイル名の前のprefixにする
            dest_key_prefix = "/".join(object_key.split("/")[:-1])

            value = state.value

            value.update(
                {
                    "SectionId": section_id,
                    "DestKeyPrefix": dest_key_prefix,
                    "SerializeRawAudio": SerializeRawAudio(serialized),
                }
            )

            state = State("RawAudio", value)

            return state

    def send(
        self,
        key_prefix: str,
        audio_id: str,
        section_id: str,
        speaker_label: str,
        serialized: str,
    ) -> str:
        saved_file_path = self._save_wav(serialized)
        object_key = f"{key_prefix}/{audio_id}/{section_id}/{speaker_label}.wav"
        try:
            self.client.upload_file(saved_file_path, str(self.dest_bucket), object_key)
        except Exception as e:
            raise e
        return object_key

    def get_destination(self) -> str:
        return str(self.dest_bucket)

    def _save_wav(self, serialized: str) -> str:
        tmp_file_path = f"/tmp/{uuid.uuid4()}"
        with open(tmp_file_path, "w") as f:
            f.write(serialized)

        return tmp_file_path


if __name__ == "__main__":
    object_key = "companies/4ce9cf31-4bae-4cf5-bc18-55ddb061d024/visits/1/1720742400_1720746000.wav.b64"
    # ???
    section_id = object_key.split("/")[-1].split(".")[0]
    print(section_id)
    # 送信先のキーを受信時のオブジェクトキーのファイル名の前のprefixにする???
    dest_key_prefix = "/".join(object_key.split("/")[:-1])
    print(dest_key_prefix)

```

- /app/deno-sample/doll/engine/dializer/src/data_transfer/redis_transfer.py
```python
import hashlib
import json

import myredis

from . import ITransfer, ReceiveData


class RedisTransfer(ITransfer):
    """Redisをデータの送受信先として用いるクラス"""

    def __init__(self):
        self.my_redis = myredis.MyRedis()

        return

    def receive(self, event: dict) -> "ReceiveData":
        try:
            audio_object_key = event["audio_object_key"]
        except json.JSONDecodeError as e:
            raise ValueError(f"event must be a valid JSON string: {e}")

        key = myredis.UploadAudioDataKey.from_redis_key_format(audio_object_key)

        # section_idは、音声データのキーのハッシュにする
        section_id = hashlib.sha256(
            key.to_redis_key_format().encode("utf-8")
        ).hexdigest()

        serialized_audio = self.my_redis.get_object(key)

        # 送信先のキーはRedis使用では現状使用しないので空文字
        dest_key_prefix = ""

        return ReceiveData(serialized_audio, section_id, dest_key_prefix)

    def send(
        self,
        key_prefix: str,
        audio_id: str,
        section_id: str,
        speaker_label: str,
        serialized: bytes,
    ) -> str:
        key = myredis.SpeakerDelizedDataKey(
            audio_id=audio_id,
            section_id=section_id,
            speaker_label=speaker_label,
        )
        try:
            key = self.my_redis.set_object(key, serialized)
            self.my_redis.publish_message(
                myredis.Channel(myredis.SD_SUCCESSED_CHANNEL),
                myredis.PublishMessage(key),
            )
        except Exception as e:
            raise e

        return key

    def get_destination(self) -> str:
        # NOTE: Interfaceに合わせているが、Redis使用では現状使用しない(?)ので適当文字
        return "redis"

```

- /app/deno-sample/doll/engine/dializer/src/components/di/di.py
```python
"""
serverlessの仕様的に一回しか実行されないはずなので、globalでdiを保持しておく
"""

from components import di
from components import env
from data_transfer import ITransfer
from data_transfer import transfer

container = di.DIContainer()

container.register(interface=env.IEnv, implementation=env.Env)
container.register(interface=ITransfer, implementation=transfer.Transfer)



def inject() -> di.DIContainer:
    return container()


if __name__ == "__main__":
    dic = inject()

    d = dic[env.IEnv]()

```

- /app/deno-sample/doll/engine/dializer/src/components/di/__init__.py
```python
"""
serverlessの仕様的に一回しか実行されないはずなので、globalでdiを保持しておく
"""

from doll_core import di
from components import env
from data_transfer import ITransfer
from data_transfer import transfer

container = di.DIContainer()

container.register(interface=env.IEnv, implementation=env.Env)
container.register(interface=ITransfer, implementation=transfer.Transfer)



dic = container.commit()


if __name__ == "__main__":

    d = dic[env.IEnv]()

```

- /app/deno-sample/doll/engine/dializer/src/components/entity/diarized_speakers.py
```python
from components.abstracts import IDiarizedSpeakers, IDiarizedSpeaker, IState
from typing import List, Dict
from components.value_objects.state import State
from typing import List, Iterator
from components.entity.diarized_speaker import DiarizedSpeaker


# ここでDiarizedSpeakersをserialize()したときに、
#  # {
# #     "CompanyUUID": "4ce9cf31-4bae-4cf5-bc18-55ddb061d024",
# #     "visitID": "1234",
# #     "destination": "dest",
# #     "objectKeys": [
# #         "companies/4ce9cf31-4bae-4cf5-bc18-55ddb061d024/visits/1/test/1720742400_1720746000/SPEAKER_00.wav",
# #         "companies/4ce9cf31-4bae-4cf5-bc18-55ddb061d024/visits/1/test/1720742400_1720746000/SPEAKER_01.wav"
# #     ]
# # }
# この形になる必要があるわけね
class DiarizedSpeakers(IDiarizedSpeakers):
    """
    Listとして振る舞うDiarizedSpeakerの集合を表すクラス。
    dializerと共通。
    """

    # entityも外部から暗黙的なプロパティ追加されることは想定しないので__slotsを定義する
    __slots__ = "_array"
    _array: List[IDiarizedSpeaker]

    def __init__(self, value: IDiarizedSpeaker) -> None:
        self._array = [value]
        pass

    def append(self, value: IDiarizedSpeaker) -> None:
        self._array.append(value)

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def serialize(self) -> State:
        data = [value.serialize().to_dict() for value in self._array]

        data = State(key=self.__class__.__name__, value=data)
        return data

    @staticmethod
    def deserialize(state: IState) -> "IDiarizedSpeakers":
        # TODO: かなり冗長に書いてるのでリファクタリングする
        if state.key != DiarizedSpeakers.__name__:
            raise ValueError(f"Expected {DiarizedSpeakers.__name__}, got {state.key}")

        values: List = state.value

        first_value = values[0]
        key_class_name: str = list(first_value.keys())[0]
        value_dializer = first_value[key_class_name]

        dializer_speaker = DiarizedSpeaker.deserialize(
            State(key=key_class_name, value=value_dializer)
        )

        dializer_speakers = DiarizedSpeakers(dializer_speaker)

        for value in values[1:]:
            key_class_name: str = list(value.keys())[0]
            value_dializer = value[key_class_name]
            dializer_speaker = DiarizedSpeaker.deserialize(
                State(key=key_class_name, value=value_dializer)
            )

            dializer_speakers.append(dializer_speaker)

        return dializer_speakers

    def to_dict(self) -> None:
        raise NotImplementedError

    def to_json(self) -> str:
        return self.serialize().to_json()

    # 以下はlistとして振る舞うために必要
    def __getitem__(self, index: int) -> IDiarizedSpeaker:
        return self._array[index]

    def __len__(self) -> int:
        return len(self._array)

    def __iter__(self) -> Iterator[IDiarizedSpeaker]:
        return iter(self._array)

    def __contains__(self, item: IDiarizedSpeaker) -> bool:
        return item in self._array

    def remove(self, item: IDiarizedSpeaker) -> None:
        self._array.remove(item)

    @property
    def array(self) -> List[IDiarizedSpeaker]:
        return self._array


if __name__ == "__main__":
    import os
    from components.utils import get_file_path, load_audio
    from components.entity.diarized_speaker import DiarizedSpeaker
    from components.value_objects.speaker_source import SpeakerSource
    import sys
    import json

    try_section_dir = "/app/sample_audio/section1"

    if not os.path.isdir(try_section_dir):
        raise f"{try_section_dir} is not a directory."

    # dirからfileのパスを取得する
    file_paths = get_file_path(try_section_dir)

    # torchAudioで読み込む
    audio_data = [load_audio(file_path=file_path) for file_path in file_paths]
    new_speakers = [
        DiarizedSpeaker(speaker_source=SpeakerSource(value=waveform, sample_rate=sr))
        for waveform, sr in audio_data
    ]

    dss = DiarizedSpeakers(new_speakers[0])
    for new_speaker in new_speakers[1:]:
        dss.append(new_speaker)

    json_data = dss.to_json()

    # シリアライズされたデータのバイトサイズを取得
    size_in_bytes = sys.getsizeof(json_data)

    # 400KBを超えるかどうかをチェック
    print(f"""
          Item size: 
          {size_in_bytes} bytes

          {size_in_bytes / 1024} KB
          
          {size_in_bytes / 1024 / 1024} MB
          
          {size_in_bytes / 1024 / 1024 / 1024} GB
          
          """)
    if size_in_bytes > 400 * 1024:  # 400KB = 400 * 1024 bytes
        print(f"❌ Item size is {size_in_bytes} bytes, which exceeds the 400KB limit.")
    else:
        print(
            f"⭕️ Item size is {size_in_bytes} bytes, which is within the 400KB limit."
        )

    with open("dialized_speakers.json", "w") as f:
        f.write(json_data)

    with open("dialized_speakers.json", "r") as f:
        data = f.read()

    response_dict: Dict = json.loads(data)

    key_class_name = list(response_dict.keys())[0]

    state = State(key=key_class_name, value=response_dict[key_class_name])

    a = DiarizedSpeakers.deserialize(state)

    print(a)

```

- /app/deno-sample/doll/engine/dializer/src/components/entity/__init__.py
```python

```

- /app/deno-sample/doll/engine/dializer/src/components/entity/diarized_speaker.py
```python
from components import const
from components.value_objects.embedding_vector import EmbeddingVector
from components.abstracts import IDiarizedSpeaker
import copy
from components.value_objects.speaker_source import SpeakerSource
from typing import Optional
from components.value_objects.state import State
from typing import Dict
from components.abstracts import ISerializable
from components.value_objects.speaker_id import SpeakerId


class DiarizedSpeaker(IDiarizedSpeaker):
    """
    音源分離されたスピーカーの情報を保持するクラス
    """

    __slots__ = ("_speaker_source", "_embedding_vector", "_tmp_speaker_id")

    # cloneして新しいaddressを作成したいのは_embedding_vectorだけでwaveformは同じaddressを使いたい。。。

    def __init__(self, speaker_source: Optional[SpeakerSource] = None):
        self._speaker_source: Optional[SpeakerSource] = speaker_source

        # 埋め込みベクトルを求める以外の方法になるかもしれない
        self._embedding_vector: Optional[EmbeddingVector] = None

        # 一時的にspeaker_idを保持する変数
        # 話者識別されるまでの一時的な変数
        self._tmp_speaker_id = SpeakerId()

    def __eq__(self, other: "DiarizedSpeaker") -> bool:
        if not isinstance(other, DiarizedSpeaker):
            return False
        return self._tmp_speaker_id == other._tmp_speaker_id

    def __str__(self) -> str:
        # TODO: そういうオーバーロードは良くないと思うなー
        return str(self._tmp_speaker_id)

    @property
    def speaker_source(self) -> Optional[SpeakerSource]:
        return getattr(self, "_speaker_source", None)

    def serialize(self) -> State:
        key = self.__class__.__name__

        if self.embedding_vector is None and self.speaker_source is not None:
            value = {
                "_speaker_source": self.speaker_source.serialize().to_dict(),
                "_embedding_vector": None,
            }
        elif self.embedding_vector is not None:
            value = {
                "_speaker_source": None,
                "_embedding_vector": self.embedding_vector.serialize().to_dict(),
            }
        else:
            raise Exception("speaker_source and embedding_vector are None.")
        state = State(key=key, value=value)
        return state

    @staticmethod
    def deserialize(state: ISerializable) -> "DiarizedSpeaker":
        class_name_from_state = state.key
        if class_name_from_state != DiarizedSpeaker.__name__:
            raise Exception("class name is not matched.")

        embedding_vector_value: Dict = state.value["_embedding_vector"]

        speaker_source_value: Dict = state.value["_speaker_source"]

        ds = DiarizedSpeaker()
        if embedding_vector_value is not None:
            class_name = EmbeddingVector.__name__
            embedding_vector = EmbeddingVector.deserialize(
                State(key=class_name, value=embedding_vector_value[class_name])
            )
            ds.embedding_vector = embedding_vector
        if speaker_source_value is not None:
            class_name = SpeakerSource.__name__
            speaker_source = SpeakerSource.deserialize(
                State(key=class_name, value=speaker_source_value[class_name])
            )
            ds._speaker_source = speaker_source

        return ds

    def to_dict(self) -> None:
        raise NotImplementedError

    def to_json(self) -> str:
        raise NotImplementedError

    @property
    def memory(self):
        if self.speaker_source is None:
            raise Exception("speaker_source is not set.")
        return const.PersonalAudioDict(
            waveform=self.speaker_source.tensor,
            sample_rate=self.speaker_source.sample_rate,
        )

    @property
    def embedding_vector(self) -> Optional[EmbeddingVector]:
        return self._embedding_vector

    @embedding_vector.setter
    def embedding_vector(self, a: EmbeddingVector) -> Optional[EmbeddingVector]:
        # 埋め込みベクトルを計算したら、speaker_sourceは不要になるので削除する
        # 後続の処理でspeaker_sourceが必要になることはなく、埋め込みベクトル同士の比較を行なっていくため。

        if self.speaker_source is not None:
            del self._speaker_source
        self._embedding_vector = a

    def clone(self) -> "DiarizedSpeaker":
        """
        # この関数は、speakerの情報をコピーして新しいspeakerを作成する
        # 非同期で処理した際に、pythonは参照渡しを行うため、speakerの情報が上書きされてしまう
        # この時、予期せぬ挙動を防ぐために、この関数を用いてspeakerの情報をコピーする
        """
        res = copy.deepcopy(self)
        return res

```

- /app/deno-sample/doll/engine/dializer/src/components/value_objects/speaker_id.py
```python
from dataclasses import dataclass
import uuid
from typing import Optional


@dataclass(frozen=True)
class SpeakerId:
    """
    immutableなデータクラスです。
    why: 非同期で処理した際に予期せぬ挙動を防ぐために、このクラスはimmutableです。
    このクラスは、speakerIDを表すクラスです。
    インスタンスが呼ばれた際に、speakerIDを一度だけ生成します。
    """

    # __dict__を生やさないようにしてメモリ消費を抑えるために、__slots__を使っている
    # value onjectは本質的にたくさん生成される可能性が高いです。そのため、__slots__を使ってメモリ消費を抑えることが重要です。
    __slots__ = "_speaker_id"
    # __slots__を使うとdefault_factoryが使えなくなります。
    _speaker_id: str

    def __new__(cls, speaker_id: Optional[str] = None) -> "SpeakerId":
        obj = super().__new__(cls)
        # ここでspeakerIDを生成する
        if not speaker_id:
            speaker_id = str(uuid.uuid4().hex)
        # インスタンス作成時に一度だけ作成するため、__setattr__を使って代入する
        object.__setattr__(obj, "_speaker_id", speaker_id)
        return obj

    # @dataclass(frozen=True)したときに__init__が自動的に作成されるが、そちらは勝手に引数を期待してしまうので、
    # それを上書きして無効にするために書いています
    def __init__(self, speaker_id: Optional[str] = None):
        pass

    def __eq__(self, other) -> bool:
        # pythonだからunreachableなことないんだけどな。。。
        # タイプヒントつけるとpylanceがraiseに到達しないとか言ってくるからこういう書き方になってしまう
        if not isinstance(other, SpeakerId):
            raise ValueError(f"{other} is not an instance of SpeakerId.")
        return self._speaker_id == other._speaker_id

    def __str__(self) -> str:
        return self._speaker_id


if __name__ == "__main__":
    speaker_id = SpeakerId()
    print(str(speaker_id))
    print(str(speaker_id))
    print(str(speaker_id))

    speaker_id = SpeakerId()
    print(str(speaker_id))

    speaker_id = SpeakerId()
    print(str(speaker_id))

    id = "test"
    speaker_id = SpeakerId(id)
    print(str(speaker_id))

```

- /app/deno-sample/doll/engine/dializer/src/components/value_objects/input_event.py
```python
from doll_core.value_objects.input_event_dict import InputEventDict
from dataclasses import dataclass
from typing import Tuple
from doll_core.value_objects.company_uuid import CompanyUUID
from doll_core.value_objects.state_key import StateKey
from doll_core.value_objects.s3_object_key import S3ObjectKey

IS_B64 = bool
COMPANY_UUID = str
VISIT_ID=str
OBJECT_KEY=str

@dataclass(frozen=True)
class InputEvent:
    """
    dializerが受け取る入力イベントを表すクラス
    """

    __slots__ = ["company_uuid", "state_key", "is_b64", "s3_object_key"]
    company_uuid: CompanyUUID
    state_key: StateKey
    is_b64: IS_B64
    s3_object_key: S3ObjectKey

    def __new__(cls, v: InputEventDict) -> "InputEvent":
        company_uuid, visit_id, is_b64, object_key  = cls._check(v)

        obj = super().__new__(cls)
        object.__setattr__(obj, "company_uuid", CompanyUUID(company_uuid))
        object.__setattr__(obj, "state_key", StateKey(visit_id))
        object.__setattr__(obj, "is_b64", is_b64)
        object.__setattr__(obj, "s3_object_key", S3ObjectKey(object_key))
        return obj

    def __init__(self, v: InputEventDict):
        pass

    @staticmethod
    def _check(v: InputEventDict) -> Tuple[COMPANY_UUID, VISIT_ID, IS_B64, OBJECT_KEY]:
        # InputEventDictに存在するかと型チェックを行う
        if "CompanyUUID" not in v:
            raise ValueError("CompanyUUID is required")
        if not isinstance(v["CompanyUUID"], COMPANY_UUID):
            raise ValueError("CompanyUUID is not str")

        company_uuid = v["CompanyUUID"]

        if "VisitID" not in v:
            raise ValueError("VisitID is required")
        if not isinstance(v["VisitID"], VISIT_ID):
            raise ValueError("VisitID is not str")

        visit_id = v["VisitID"]

        if "IsB64" not in v:
            raise ValueError("IsB64 is required")
        if not isinstance(v["IsB64"], IS_B64):
            raise ValueError("IsB64 is not bool")

        is_b64 = v["IsB64"]

        if "ObjectKey" not in v:
            raise ValueError("ObjectKey is required")
        if not isinstance(v["ObjectKey"], OBJECT_KEY):
            raise ValueError("ObjectKey is not str")
        
        object_key = v["ObjectKey"]

        return (
            company_uuid,
            visit_id,
            is_b64,
            object_key
        )


if __name__ == "__main__":
    e = InputEvent(
        InputEventDict(
            {
                "CompanyUUID": "company_uuid",
                "VisitID": "visit_id",
                "IsB64": True,
                "ObjectKey": "object_key"
            }
        )
    )
    print(e.company_uuid)
    print(e.state_key)
    print(e.is_b64)
    print(e.s3_object_key)

```

- /app/deno-sample/doll/engine/dializer/src/components/value_objects/speaker_source.py
```python
from dataclasses import dataclass
import numpy as np
from torch import Tensor
from components.abstracts import IRefSerializable
import gzip
import weakref
from components.value_objects.state import State
from components.abstracts import IState
from typing import Dict
import torch
from components.logger import Logger
from logging import Logger as logging_logger


@dataclass(frozen=True)
class SpeakerSource(IRefSerializable):
    """

    immutableなデータクラスです。
    why: 非同期で処理した際に予期せぬ挙動を防ぐために、このクラスはimmutableです。

    このクラスは、話者分離アルゴリズムによって分離された音声データをカプセル化します。
    音声データは、torch.tensor または np.ndarray , Noneとして提供され、用途に応じて柔軟に扱うことができます。

    また、弱参照で参照するため、メモリ消費を抑えることができます。

    """

    # __dict__を生やさないようにしてメモリ消費を抑えるために、__slots__を使っている
    # value onjectは本質的にたくさん生成される可能性が高いです。そのため、__slots__を使ってメモリ消費を抑えることが重要です。
    __slots__ = ("_value", "_sample_rate", "_logger")
    _value: np.ndarray | Tensor
    _sample_rate: int
    _logger: logging_logger

    def __new__(cls, value: np.ndarray | Tensor, sample_rate: int) -> "SpeakerSource":
        obj = super().__new__(cls)
        logger = Logger.init(f"{__file__}:{__name__}")

        # インスタンス作成時に一度だけ作成するため、__setattr__を使って代入する
        object.__setattr__(obj, "_value", value)
        object.__setattr__(obj, "_sample_rate", sample_rate)
        object.__setattr__(obj, "_logger", logger)

        return obj

    # @dataclass(frozen=True)したときに__init__が自動的に作成されるが、そちらは勝手に引数を期待してしまうので、
    # それを上書きして無効にするために書いています
    def __init__(self, value: np.ndarray | Tensor, sample_rate: int) -> None:
        pass

    def ref(self) -> "SpeakerSource":
        """
        弱参照を返す。
        弱参照は、参照カウントを増やさずにオブジェクトを参照することができる。

        戻り値が"SpeakerSource"となっていますが嘘です。本当は"weakref.ReferenceType"です。
        なぜこうしているのかというとtype hintingが殺されてしまうからです。。。
        くそうpythonめ。。。

        弱参照使うとパフォーマンス追い求めれますが予期せぬガベージコレクションが発生する可能性があるので注意してください。。
        c/c++でありがちなやつ、最近(2024/08/15周辺)だとクラウドストライクのあれです。(めっちゃおもろいので興味ある人は調べてね。)
        rustならこういう問題起きないんだけどな。。。
        rustのスマートポインタが恋しい。。

        merit:
        参照カウンタが増えないため、メモリを節約できる。
        また、del hoge みたいにすることで、オブジェクトを削除扱い(※諸説あり。元々占有してたメモリをほぼほぼ解放するからまあ。。。)にすることができる。

        demerit:
        参照カウンタが増えないため、予期せぬタイミングでガベージコレクションが発生する可能性がある。
        例えば、予期せぬタイミングで参照カウンタが0になったときにガベージコレクションが発生することがある。

        why:
        話者分離アルゴリズムによって分離された音声データのテンソルはそもそも大きなオブジェクトである。
        それがずっと保持されるということは線形、最悪には指数的にメモリを消費することになる。
        具体的には、会議が数時間と長丁場になったとすると、その分生じるテンソルの数も増えるし、カウントされる話者が一人増えるたびに組み合わせがn通り増えることになる。
        これをずっと保持し続けると、メモリを圧迫することになり、プログラムが予期せぬクラッシュをしたり、よくわからないスケールをしたりする可能性がある。
        これを防ぐために、弱参照を使って、参照カウンタを増やさずオブジェクトを参照することができるようにしている。
        """
        rfr = weakref.ref(self)
        return rfr()

    def __del__(self):
        self._logger.debug(f"SpeakerSource:{id(self)} is deleted.")

    @property
    def value(self) -> np.ndarray | Tensor:
        return self._value

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def ndarray(self) -> np.ndarray:
        if isinstance(self._value, np.ndarray):
            return self._value
        raise ValueError("not np.ndarray")

    @property
    def tensor(self) -> Tensor:
        if isinstance(self._value, Tensor):
            return self._value
        raise ValueError("not Tensor")

    def serialize(self) -> IState:
        """
        SpeakerSourceをシリアライズする。
        重いから頻繁に叩かないこと。
        """
        key = self.__class__.__name__

        if isinstance(self._value, Tensor):
            data = self._value.numpy().tobytes()
        else:
            data = self._value.tobytes()

        gzip_data = gzip.compress(data).hex()
        data = {
            "sample_rate": self._sample_rate,
            "shape": self._value.shape,
            "dtype": str(self._value.dtype),
            "data": gzip_data,
            "is_tensor": isinstance(self._value, Tensor),
            "is_ndarray": isinstance(self._value, np.ndarray),
        }

        state = State(key=key, value=data)
        return state

    @staticmethod
    def deserialize(state: IState) -> "SpeakerSource":
        # np.ndarray | Tensorとして解釈可能かどうかをチェックする
        class_name_from_state = state.key
        if class_name_from_state != SpeakerSource.__name__:
            raise Exception("class name is not matched.")

        state_value: Dict = state.value
        sample_rate = state_value["sample_rate"]

        gzipped_value = bytes.fromhex(state_value["data"])
        decompressed_value = gzip.decompress(gzipped_value)
        # state_value["dtype"] に対応する torch のデータ型を取得
        dtype_str = state_value["dtype"]
        dtype: np.float32 = None
        if dtype_str == "torch.float32":
            dtype = np.float32
        is_tensor = state_value["is_tensor"]
        if is_tensor:
            shape = tuple(state_value["shape"])
            # dtype_str を適切な torch のデータ型に変換
            value = torch.from_numpy(
                np.frombuffer(decompressed_value, dtype=dtype).reshape(shape)
            )
            return SpeakerSource(value, sample_rate)

        is_ndarray = state_value["is_ndarray"]
        if is_ndarray:
            shape = tuple(state_value["shape"])
            dtype = np.dtype(state_value["dtype"])
            value = np.frombuffer(decompressed_value, dtype=dtype).reshape(shape)
            return SpeakerSource(value, sample_rate)

    def to_dict(self) -> None:
        raise NotImplementedError

    def to_json(self) -> str:
        raise NotImplementedError


if __name__ == "__main__":

    def print_size(obj):
        from pympler import asizeof

        size_in_bytes = asizeof.asizeof(obj)
        size_in_gb = size_in_bytes / (1024**3)
        return f"Size in GB: {size_in_gb:.10f} GB"

    sr = 16000

    ss = SpeakerSource(
        np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), sr
    ).ref()

    state = ss.serialize()

    a = ss.deserialize(state)

    print(a)

    del ss

```

- /app/deno-sample/doll/engine/dializer/src/components/value_objects/embedding_vector.py
```python
import numpy as np
from dataclasses import dataclass
from components.abstracts import ISerializable
import gzip
import json
from components.value_objects.state import State
from typing import Dict


@dataclass(frozen=True)
class EmbeddingVector(ISerializable):
    """
    埋め込みベクトルを表すクラスです。
    """

    # TODO: 計算元の音声データのサイズを保持しておく。
    # それを保持しておくことで、最も特徴量を表す話者を選択する際に、使用する。

    __slots__ = "value"
    value: np.ndarray

    def __new__(cls, value: np.ndarray) -> "EmbeddingVector":
        obj = super().__new__(cls)
        cls._check(value)
        object.__setattr__(obj, "value", value)
        return obj

    @staticmethod
    def _check(value: np.ndarray):
        """
        形状は(512,)であることを確認します。
        """
        if value.shape != (512,):
            raise ValueError(
                f"The shape of the embedding vector must be (512,), but got {value.shape}"
            )

    def serialize(self) -> State:
        key = self.__class__.__name__

        value = self.value.tobytes()
        dtype = str(self.value.dtype)
        gzipped_value = gzip.compress(value).hex()
        data = {
            "value": gzipped_value,
            "dtype": dtype,
            "shape": self.value.shape,
        }

        state = State(key=key, value=data)

        return state

    @staticmethod
    def deserialize(state: ISerializable) -> "EmbeddingVector":
        class_name_from_state = state.key
        if class_name_from_state != EmbeddingVector.__name__:
            raise Exception("class name is not matched.")

        state_value: Dict = state.value
        gzipped_value = bytes.fromhex(state_value["value"])
        decompressed_value = gzip.decompress(gzipped_value)
        shape = tuple(state_value["shape"])  # 形状を読み込み
        dtype = np.dtype(state_value["dtype"])  # dtypeを読み込み

        value = np.frombuffer(decompressed_value, dtype=dtype).reshape(shape)

        return EmbeddingVector(value=value)

    def to_dict(self) -> None:
        raise NotImplementedError

    def to_json(self) -> str:
        raise NotImplementedError

```

- /app/deno-sample/doll/engine/dializer/src/components/value_objects/__init__.py
```python

```

- /app/deno-sample/doll/engine/dializer/src/components/value_objects/context.py
```python



from dataclasses import dataclass
from components.value_objects.input_event import InputEvent
from doll_core.aws_s3 import CHK_RAW_FILE_S3_BUCKET
from typing import Any

@dataclass(frozen=True)
class Context:
    __slots__ = ["input_event", "src_bucket_name", "aws_context"]
    input_event: InputEvent
    aws_context: Any
    src_bucket_name = CHK_RAW_FILE_S3_BUCKET
    

    def __new__(cls, _input_event: InputEvent, _aws_context: Any) -> "Context":
        obj = super().__new__(cls)

        object.__setattr__(obj, "input_event", _input_event)
        object.__setattr__(obj, "aws_context", _aws_context)
        return obj

    def __eq__(self, other) -> bool:
        if not isinstance(other, Context):
            raise ValueError(f"{other} is not an instance of Context.")
        return self._value == other._value

    def __str__(self) -> str:
        return self._value

```

- /app/deno-sample/doll/engine/dializer/src/components/__init__.py
```python

```

- /app/deno-sample/doll/engine/dializer/src/components/utils/__init__.py
```python
from components import const
from typing import List, Tuple, Literal
import os
import torchaudio
import torch
import sys

STATE_STORE_SERVICE_VALUE = Literal[
    "DIARIZER_STATE_STORE_SERVICE", "IDENTIFIER_STATE_STORE_SERVICE"
]


def get_state_store_service(
    value: STATE_STORE_SERVICE_VALUE,
) -> const.StateStoreType:
    val = os.getenv(value)
    if val is None:
        raise f"{value} is not defined"

    try:
        return const.StateStoreType(val)
    except ValueError:
        raise f"{value} is invalid. {val}"


def get_file_path(dir_path: const.DirPath) -> List[const.DiarizedWavPath]:
    file_paths: List[const.DiarizedWavPath] = []

    for root, _, files in os.walk(dir_path):
        for filename in files:
            # パスを作成
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


def load_audio(
    file_path: const.DiarizedWavPath,
) -> Tuple[torch.Tensor, const.SampleRate]:
    return torchaudio.load(file_path)



```

- /app/deno-sample/doll/engine/dializer/src/components/env/__init__.py
```python
import os
from typing import Protocol
from components.utils import get_state_store_service
from components import const
from dataclasses import dataclass


class IEnv(Protocol):
    DIARIZER_STATE_STORE_SERVICE: const.StateStoreType

    # redis
    REDIS_HOST: str
    REDIS_PORT: str
    REDIS_CHANNNEL: str

    # aws (common)
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION_NAME: str

    # aws s3(minio)
    AWS_S3_ENDPOINT_URL: str


@dataclass(frozen=True)
class Env(IEnv):
    """
    動的にセットされる全ての環境変数を保持するイミュータブルクラス
    """

    DIARIZER_STATE_STORE_SERVICE: const.StateStoreType

    # redis
    REDIS_HOST: str
    REDIS_PORT: str
    REDIS_CHANNNEL: str

    # aws (common)
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION_NAME: str

    # aws s3(minio)
    AWS_S3_ENDPOINT_URL: str

    def __new__(cls) -> "Env":
        obj = super().__new__(cls)
        _dializer_state_store_service = get_state_store_service(
            "DIARIZER_STATE_STORE_SERVICE"
        )

        # redis
        _redis_host = os.getenv("REDIS_HOST") or ""
        _redis_port = os.getenv("REDIS_PORT") or ""
        _redis_channnel = os.getenv("REDIS_CHANNNEL") or ""

        # aws (common)
        _aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID") or ""
        _aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY") or ""
        _aws_region_name = os.getenv("AWS_REGION_NAME") or ""

        # s3(minio)
        _aws_s3_endpoint_url = os.getenv("AWS_S3_ENDPOINT_URL") or ""

        object.__setattr__(
            obj, "DIARIZER_STATE_STORE_SERVICE", _dializer_state_store_service
        )
        # redis
        object.__setattr__(obj, "REDIS_HOST", _redis_host)
        object.__setattr__(obj, "REDIS_PORT", _redis_port)
        object.__setattr__(obj, "REDIS_CHANNNEL", _redis_channnel)

        # aws (common)
        object.__setattr__(obj, "AWS_ACCESS_KEY_ID", _aws_access_key_id)
        object.__setattr__(obj, "AWS_SECRET_ACCESS_KEY", _aws_secret_access_key)
        object.__setattr__(obj, "AWS_REGION_NAME", _aws_region_name)

        # s3(minio)
        object.__setattr__(obj, "AWS_S3_ENDPOINT_URL", _aws_s3_endpoint_url)

        return obj

    def __init__(self) -> None:
        pass


if __name__ == "__main__":
    env = Env()
    print(env.STATE_STORE_SERVICE)
    print(env.REDIS_HOST)
    pass

```

- /app/deno-sample/doll/engine/dializer/src/components/const/__init__.py
```python
from enum import Enum
import torch
from typing import List, Tuple, TypedDict, Dict


DirPath = str
DiarizedWavPath = str
SampleRate = int

# 音源分離後の音声ファイルを切り捨てる閾値（秒）
MIN_SPEECH_DURATION_SEC = 8

# 音声分離モデルのパス
OFFLINE_MODEL_PATH = "/app/src/speaker_dialization/pyannote-audio-config/config.yaml"


class StateStoreType(Enum):
    REDIS = "REDIS"
    S3 = "S3"
    DYNAMODB = "DYNAMODB"
    S3_MINIO = "S3_MINIO"


class PersonalAudioDict(TypedDict):
    waveform: torch.Tensor
    sample_rate: int

```

- /app/deno-sample/doll/engine/dializer/src/components/abstracts/__init__.py
```python
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
from components import const


# シリアライズ可能なクラスのためのインターフェース
class ISerializable(ABC):
    @abstractmethod
    def serialize(self) -> "IState":
        pass

    @staticmethod
    @abstractmethod
    def deserialize(serialized: "IState") -> "IState":
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        pass

    @abstractmethod
    def to_json(self) -> str:
        pass


class IState(ISerializable):
    @property
    @abstractmethod
    def key(self) -> str:
        pass

    @property
    @abstractmethod
    def value(self) -> Dict:
        pass

    pass


class IRefSerializable(ISerializable):
    @abstractmethod
    def ref(self) -> "IRefSerializable":
        """
        弱参照を返す。
        """
        pass

    @abstractmethod
    def __del__(self):
        pass


class ISpeaker(ISerializable):
    @property
    @abstractmethod
    def memory(self) -> "const.PersonalAudioDict":
        pass

    @property
    @abstractmethod
    def embedding_vector(self) -> ISerializable:
        pass

    @embedding_vector.setter
    @abstractmethod
    def embedding_vector(self, a: ISerializable) -> None:
        pass

    @abstractmethod
    def clone(self) -> "ISpeaker":
        pass


class IDiarizedSpeaker(ISpeaker):
    @property
    @abstractmethod
    def speaker_source(self) -> Optional[IRefSerializable]:
        pass


class IDiarizedSpeakers(ISerializable):
    @abstractmethod
    def append(self, value: IDiarizedSpeaker) -> None:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @property
    @abstractmethod
    def array(self) -> List[IDiarizedSpeaker]:
        pass

```

- /app/deno-sample/doll/engine/dializer/src/audio/raw_audio.py
```python
import audio
import base64
import io
import torchaudio


def tensol_audio_data(data: audio.B64EncodedRawAudio):
    # base64_decodeする
    base64_decoded_audio_data = base64.b64decode(data.raw_chunk)

    # BytesIOオブジェクトにデータを格納
    audio_buffer = io.BytesIO(base64_decoded_audio_data)
    waveform, sample_rate = torchaudio.load(audio_buffer)

    return waveform, sample_rate


class RawAudio:
    def __init__(self, audio: audio.B64EncodedRawAudio):
        self.audio = audio
        self.waveform, self.sample_rate = self.to_tensor()

    def to_tensor(self):
        return tensol_audio_data(self.audio)

    def in_memmory(self):
        audio_in_memory = {"waveform": self.waveform, "sample_rate": self.sample_rate}
        return audio_in_memory

```

- /app/deno-sample/doll/engine/dializer/src/audio/__init__.py
```python
from pydub import AudioSegment
import base64
import gzip
import json
import io

Base64EncodedAudio = str
SerializeRawAudio = bytes
"""
gzip compressed json string
{
    "b64encode_audio_str": str,
    "chunk_length_ms": int,
    "audio_id": str
}
"""


def load_audio(file_path: str) -> AudioSegment:
    # Load audio file
    audio = AudioSegment.from_file(file_path)

    return audio


class B64EncodedRawAudio:
    """
    publishされるものよねこれ
    """

    def __init__(self, chunk: AudioSegment | None, chunk_length_ms: int, audio_id: str):
        if chunk is not None:
            self.raw_chunk = base64.b64encode(self._buffer(chunk)).decode("utf-8")
        else:
            self.raw_chunk = None

        self.chunk_length_ms = chunk_length_ms
        self.audio_id = audio_id

    def _buffer(self, chunk: AudioSegment) -> bytes:
        # AudioSegmentオブジェクトをバイナリデータに変換する
        buffer = io.BytesIO()
        chunk.export(buffer, format="wav")
        return buffer.getvalue()
    
    


    def to_serialize(self) -> SerializeRawAudio:
        serialized = gzip.compress(
            json.dumps(
                {
                    # TODO: b64エンコードしてる
                    "b64encode_audio_str": self.raw_chunk,
                    # TODO: 45000ms固定値
                    "chunk_length_ms": self.chunk_length_ms,
                    # TODO: 一意のID: s3のkey
                    "audio_id": self.audio_id,
                }
            ).encode("utf-8")
        )
        return serialized

    @classmethod
    def from_serialize(cls, serialized: SerializeRawAudio) -> "B64EncodedRawAudio":
        decompressed = gzip.decompress(serialized)
        decompressed = json.loads(decompressed)
        print(type(decompressed))

        c = cls(
            chunk=None,
            chunk_length_ms=decompressed["chunk_length_ms"],
            audio_id=decompressed["audio_id"],
        )

        c.raw_chunk = decompressed["b64encode_audio_str"]

        return c

```

- /app/deno-sample/doll/engine/dializer/src/usecase/__init__.py
```python

```

- /app/deno-sample/doll/engine/dializer/src/dializer/__init__.py
```python
# DIコンテナの取得
from components.di import dic
from doll_core.logger import Logger
from doll_core.value_objects.input_event_dict import InputEventDict
from components.value_objects.input_event import InputEvent
from components.value_objects.context import Context


logger = Logger.init(f"{__file__}:{__name__}")

def dializer(input_event, contex):
    logger.debug(f"event type: {type(input_event)}")
    input_event_dict = InputEventDict(input_event)
    dializer_event = InputEvent(input_event_dict)
    
    d_ctx = Context(input_event=dializer_event, aws_context=contex)
```

- /app/deno-sample/doll/engine/dializer/src/speaker_dialization/lib.py
```python
import torch


def scale_audio_data(audio_data: torch.Tensor) -> torch.Tensor:
    """
    音声データをスケーリングする
    モノにもする。（ここに置くものではなさそう。。。？）
    """
    audio_data = (audio_data - audio_data.mean()) / audio_data.std()
    audio_data = audio_data[0:1, :]
    return audio_data

```

- /app/deno-sample/doll/engine/dializer/src/speaker_dialization/__init__.py
```python
import audio
import audio.raw_audio
import torch
from typing import List, Dict, Optional
from components import const
from speaker_dialization.lib import scale_audio_data
from components.entity.diarized_speakers import DiarizedSpeakers
from components.entity.diarized_speaker import DiarizedSpeaker
from components.value_objects.speaker_source import SpeakerSource
from components.logger import Logger
import os

SPEAKER_LABEL = str


class PyAnnoteAudio:
    def __init__(self) -> None:
        from pyannote.audio import Pipeline

        self.logger = Logger.init(class_name=self.__class__.__name__)
        self.logger.debug("PyAnnoteAudio init")
        self.logger.debug(f"offline model path: {const.OFFLINE_MODEL_PATH}")
        cache_dir = os.getenv("HF_HUB_CACHE")
        self.logger.debug(f"cache_dir: {cache_dir}")
        # ※環境変数もcache_dirも指定しないと外に取りに行こうとしてしまう
        self.pipeline = Pipeline.from_pretrained(checkpoint_path=const.OFFLINE_MODEL_PATH, cache_dir=cache_dir)


class SpeakerDialization:
    def __init__(self):
        self.pyannote_audio = PyAnnoteAudio()
        self._diarized_speakers: Optional[DiarizedSpeakers] = None

    @property
    def diarized_speakers(self) -> DiarizedSpeakers:
        if self._diarized_speakers is None:
            raise ValueError("diarized_speakers is required")
        return self._diarized_speakers

    def prune_short_audio(self, threshold: int):
        """
        閾値以下の音声データを削除する
        """
        d_speakers = self.diarized_speakers

        def is_short(speaker: DiarizedSpeaker) -> bool:
            sample_rate = speaker.speaker_source.sample_rate
            min_samples = int(threshold * sample_rate)
            if speaker.speaker_source.tensor.shape[1] < min_samples:
                return True
            return False

        for d_speaker in d_speakers:
            if is_short(d_speaker):
                d_speakers.remove(d_speaker)

        self._diarized_speakers = d_speakers

    # TODO サービスかなあ
    def _scale(self, d_speaker: DiarizedSpeaker) -> "DiarizedSpeaker":
        """
        音声データをスケールします
        """
        speaker_waveform = scale_audio_data(d_speaker.speaker_source.tensor)

        speaker_source = SpeakerSource(
            value=speaker_waveform, sample_rate=d_speaker.speaker_source.sample_rate
        )

        d_speaker = DiarizedSpeaker(speaker_source=speaker_source)
        return d_speaker

    def _cat(
        self,
        d_speakers_by_label: Dict[str, DiarizedSpeaker],
        speaker_label: str,
        speaker_waveform: torch.Tensor,
        sample_rate: int,
    ):
        """
        labelごとに集計する
        """

        def _common(
            d_speakers_by_label: Dict[str, DiarizedSpeaker],
            speaker_label: str,
            speaker_waveform: torch.Tensor,
            sample_rate: int,
        ):
            diarized_speaker = DiarizedSpeaker(
                speaker_source=SpeakerSource(
                    value=speaker_waveform, sample_rate=sample_rate
                )
            )
            d_speakers_by_label[speaker_label] = diarized_speaker
            return d_speakers_by_label

        if speaker_label not in d_speakers_by_label:
            d_speakers_by_label = _common(
                d_speakers_by_label, speaker_label, speaker_waveform, sample_rate
            )
        else:
            d_speaker = d_speakers_by_label[speaker_label]
            labeled_waveform = d_speaker.speaker_source.tensor
            merged_waveform = torch.cat((labeled_waveform, speaker_waveform), dim=1)
            speaker_source = SpeakerSource(
                value=merged_waveform, sample_rate=sample_rate
            )
            d_speaker = DiarizedSpeaker(speaker_source=speaker_source)
            d_speakers_by_label[speaker_label] = d_speaker

        return d_speakers_by_label

    def exec(self, data: audio.raw_audio.RawAudio):
        # ここでspeaker_dializationを実行する
        audio_in_memory = data.in_memmory()
        sample_rate: int = audio_in_memory["sample_rate"]
        waveform: torch.Tensor = audio_in_memory["waveform"]

        # speaker_dializationを実行
        diarization = self.pyannote_audio.pipeline(audio_in_memory)
        # ラベルごとにここで集計すればデータロス減るじゃん
        d_speakers_by_label: Dict[str, DiarizedSpeaker] = {}

        for segment, _, speaker_label in diarization.itertracks(yield_label=True):
            start = int(segment.start * sample_rate)
            end = int(segment.end * sample_rate)
            speaker_waveform = waveform[:, start:end]
            d_speakers_by_label = self._cat(
                d_speakers_by_label, speaker_label, speaker_waveform, sample_rate
            )

        # serviceとかの処理だと思うなーこれ
        items = [
            (
                speaker_label,
                speaker,
            )
            for speaker_label, speaker in d_speakers_by_label.items()
        ]
        first = items.pop()
        first_d_speaker = self._scale(first[1])
        # scaleする
        diarized_speakers = DiarizedSpeakers(first_d_speaker)
        for item in items:
            d_speaker = self._scale(item[1])
            diarized_speakers.append(d_speaker)

        self._diarized_speakers = diarized_speakers

```
