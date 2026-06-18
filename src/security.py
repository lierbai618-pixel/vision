"""
安全模块

提供输入验证、令牌管理等安全功能
"""

import re
import hmac
import hashlib
import json
import time
import secrets
from typing import Dict, Optional, Any


class InputValidator:
    """输入验证器"""

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        验证邮箱格式

        Args:
            email: 邮箱地址

        Returns:
            是否有效
        """
        if not email or not isinstance(email, str):
            return False
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_username(username: str) -> bool:
        """
        验证用户名

        规则：3-32个字符，只能包含字母、数字、下划线

        Args:
            username: 用户名

        Returns:
            是否有效
        """
        if not username or not isinstance(username, str):
            return False
        if len(username) < 3 or len(username) > 32:
            return False
        pattern = r'^[a-zA-Z0-9_]+$'
        return bool(re.match(pattern, username))

    @staticmethod
    def validate_password(password: str) -> Dict:
        """
        验证密码强度

        规则：至少8个字符，包含大小写字母和数字

        Args:
            password: 密码

        Returns:
            验证结果，包含 valid 和 message
        """
        if not password or not isinstance(password, str):
            return {'valid': False, 'message': '密码不能为空'}

        if len(password) < 8:
            return {'valid': False, 'message': '密码至少8个字符'}

        if not re.search(r'[a-z]', password):
            return {'valid': False, 'message': '密码需包含小写字母'}

        if not re.search(r'[A-Z]', password):
            return {'valid': False, 'message': '密码需包含大写字母'}

        if not re.search(r'[0-9]', password):
            return {'valid': False, 'message': '密码需包含数字'}

        return {'valid': True, 'message': '密码有效'}

    @staticmethod
    def sanitize_input(text: str, max_length: int = 1000) -> str:
        """
        清理输入文本

        Args:
            text: 输入文本
            max_length: 最大长度

        Returns:
            清理后的文本
        """
        if not text or not isinstance(text, str):
            return ''
        # 移除控制字符
        text = ''.join(c for c in text if c.isprintable() or c in '\n\r\t')
        # 截断
        return text[:max_length]

    @staticmethod
    def validate_image_path(path: str) -> bool:
        """
        验证图片路径安全性

        Args:
            path: 文件路径

        Returns:
            是否安全
        """
        if not path or not isinstance(path, str):
            return False
        # 检查路径遍历
        if '..' in path:
            return False
        # 检查扩展名
        allowed = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        from pathlib import Path
        return Path(path).suffix.lower() in allowed


class TokenManager:
    """
    令牌管理器

    使用 HMAC-SHA256 生成和验证令牌

    Attributes:
        secret_key: 密钥
        token_expiry: 令牌有效期（秒）
    """

    def __init__(self, secret_key: Optional[str] = None, token_expiry: int = 3600):
        """
        初始化令牌管理器

        Args:
            secret_key: 密钥，None 则自动生成
            token_expiry: 令牌有效期（秒），默认 1 小时
        """
        self.secret_key = secret_key or secrets.token_hex(32)
        self.token_expiry = token_expiry

    def generate_token(self, data: Dict[str, Any]) -> str:
        """
        生成令牌

        Args:
            data: 要编码的数据

        Returns:
            令牌字符串
        """
        payload = {
            'data': data,
            'exp': time.time() + self.token_expiry,
            'iat': time.time()
        }
        payload_json = json.dumps(payload, sort_keys=True)
        payload_b64 = self._base64_encode(payload_json)

        signature = self._sign(payload_b64)
        return f"{payload_b64}.{signature}"

    def verify_token(self, token: str) -> Optional[Dict]:
        """
        验证令牌

        Args:
            token: 令牌字符串

        Returns:
            令牌数据，无效返回 None
        """
        try:
            parts = token.split('.')
            if len(parts) != 2:
                return None

            payload_b64, signature = parts

            # 验证签名
            expected_signature = self._sign(payload_b64)
            if not hmac.compare_digest(signature, expected_signature):
                return None

            # 解析 payload
            payload_json = self._base64_decode(payload_b64)
            payload = json.loads(payload_json)

            # 检查过期
            if payload.get('exp', 0) < time.time():
                return None

            return payload.get('data')

        except Exception:
            return None

    def _sign(self, data: str) -> str:
        """生成 HMAC 签名"""
        return hmac.new(
            self.secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()

    @staticmethod
    def _base64_encode(data: str) -> str:
        """Base64 编码"""
        import base64
        return base64.urlsafe_b64encode(data.encode()).decode()

    @staticmethod
    def _base64_decode(data: str) -> str:
        """Base64 解码"""
        import base64
        return base64.urlsafe_b64decode(data.encode()).decode()


if __name__ == '__main__':
    # 测试代码
    print("安全模块测试")

    # 输入验证
    print(f"邮箱验证: {InputValidator.validate_email('test@example.com')}")
    print(f"用户名验证: {InputValidator.validate_username('test_user')}")
    print(f"密码验证: {InputValidator.validate_password('Password123')}")

    # 令牌管理
    manager = TokenManager()
    token = manager.generate_token({'user_id': '123'})
    print(f"生成令牌: {token[:30]}...")

    data = manager.verify_token(token)
    print(f"验证令牌: {data}")
