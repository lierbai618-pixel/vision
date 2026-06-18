"""
认证管理模块

提供用户注册、登录、会话管理功能
"""

import json
import hashlib
import secrets
import time
from pathlib import Path
from typing import Dict, Optional


class AuthManager:
    """
    认证管理器

    基于 JSON 文件的简单用户认证系统

    Attributes:
        auth_file: 认证数据文件路径
    """

    def __init__(self, auth_file: str = 'auth.json'):
        """
        初始化认证管理器

        Args:
            auth_file: 认证数据文件路径
        """
        self.auth_file = Path(auth_file)
        self.users = {}
        self.sessions = {}
        self._load()

    def _load(self):
        """加载认证数据"""
        if self.auth_file.exists():
            try:
                with open(self.auth_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.users = data.get('users', {})
                self.sessions = data.get('sessions', {})
            except (json.JSONDecodeError, Exception) as e:
                print(f"警告: 加载认证文件失败: {e}")
                self.users = {}
                self.sessions = {}

    def _save(self):
        """保存认证数据"""
        self.auth_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'users': self.users,
            'sessions': self.sessions
        }
        with open(self.auth_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _hash_password(self, password: str, salt: str = None) -> tuple:
        """
        哈希密码

        Args:
            password: 明文密码
            salt: 盐值，None 则生成新盐

        Returns:
            (哈希值, 盐值)
        """
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return hashed, salt

    def register(self, username: str, email: str, password: str) -> Dict:
        """
        注册用户

        Args:
            username: 用户名
            email: 邮箱
            password: 密码

        Returns:
            注册结果，包含 success 和 message
        """
        # 验证输入
        if not username or len(username) < 3:
            return {'success': False, 'message': '用户名至少3个字符'}

        if not email or '@' not in email:
            return {'success': False, 'message': '邮箱格式无效'}

        if not password or len(password) < 6:
            return {'success': False, 'message': '密码至少6个字符'}

        # 检查用户名是否已存在
        if username in self.users:
            return {'success': False, 'message': '用户名已存在'}

        # 创建用户
        hashed, salt = self._hash_password(password)
        self.users[username] = {
            'email': email,
            'password_hash': hashed,
            'salt': salt,
            'created_at': time.time(),
            'last_login': None
        }

        self._save()
        return {'success': True, 'message': '注册成功'}

    def login(self, username: str, password: str) -> Dict:
        """
        用户登录

        Args:
            username: 用户名
            password: 密码

        Returns:
            登录结果，包含 success, message, session_id
        """
        if username not in self.users:
            return {'success': False, 'message': '用户名或密码错误'}

        user = self.users[username]
        hashed, _ = self._hash_password(password, user['salt'])

        if hashed != user['password_hash']:
            return {'success': False, 'message': '用户名或密码错误'}

        # 创建会话
        session_id = secrets.token_hex(32)
        self.sessions[session_id] = {
            'username': username,
            'created_at': time.time(),
            'last_active': time.time()
        }

        # 更新最后登录时间
        self.users[username]['last_login'] = time.time()
        self._save()

        return {'success': True, 'message': '登录成功', 'session_id': session_id}

    def get_user(self, session_id: str) -> Optional[Dict]:
        """
        获取用户信息

        Args:
            session_id: 会话 ID

        Returns:
            用户信息，会话无效返回 None
        """
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        username = session['username']

        if username not in self.users:
            return None

        user = self.users[username]
        session['last_active'] = time.time()

        return {
            'username': username,
            'email': user['email'],
            'created_at': user['created_at'],
            'last_login': user['last_login']
        }

    def logout(self, session_id: str) -> Dict:
        """
        用户登出

        Args:
            session_id: 会话 ID

        Returns:
            登出结果
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save()
            return {'success': True, 'message': '登出成功'}
        return {'success': False, 'message': '会话不存在'}

    def validate_session(self, session_id: str) -> bool:
        """
        验证会话是否有效

        Args:
            session_id: 会话 ID

        Returns:
            会话是否有效
        """
        return session_id in self.sessions

    def cleanup_sessions(self, max_age: int = 86400) -> int:
        """
        清理过期会话

        Args:
            max_age: 会话最大存活时间（秒），默认 24 小时

        Returns:
            清理的会话数量
        """
        now = time.time()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session['last_active'] > max_age
        ]

        for sid in expired:
            del self.sessions[sid]

        if expired:
            self._save()

        return len(expired)


if __name__ == '__main__':
    # 测试代码
    print("认证管理器测试")
    auth = AuthManager('test_auth_temp.json')

    # 注册
    result = auth.register('testuser', 'test@example.com', 'password123')
    print(f"注册: {result}")

    # 登录
    result = auth.login('testuser', 'password123')
    print(f"登录: {result}")

    if result['success']:
        # 获取用户
        user = auth.get_user(result['session_id'])
        print(f"用户信息: {user}")

        # 登出
        logout_result = auth.logout(result['session_id'])
        print(f"登出: {logout_result}")

    # 清理
    Path('test_auth_temp.json').unlink(missing_ok=True)
