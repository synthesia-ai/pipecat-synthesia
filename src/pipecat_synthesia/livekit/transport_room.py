"""Direct join: the avatar joins the transport room, the room the bot's `LiveKitTransport` is already in.

Everything knowable before the call starts is checked in `__init__` (*misuse vs session errors*): the
transport, the bot token's top-level `kind="agent"` claim (*agent-kind requirement*), a provided avatar
token against the bot identity and room that the bot token names, and the URL the avatar worker is told.

The room is borrowed (*borrowed-room rule*). `launch()` does three things:

1. rejects a bot token that has expired, unless the bot has already connected with it: such a token cannot
   connect, while the avatar worker's join, on its own token, would still succeed;
2. launches the avatar while Pipecat's own connect is still running (*connect overlap*), through
   `LiveKitAvatarSession.launch_while_connecting()`. The session request is made for the bot identity and room
   the bot token names, so nothing waits for the connect, and nothing touches `client.room` before it (see
   `_transport_access`). A bot token that names no room leaves only the connected room to say which, so the
   request then waits for the connect;
3. applies the avatar unsubscribe, the automatic half of the *self-hearing guard*: it unsubscribes the
   bot from the avatar's audio now and on every later subscription, and from its video, which the bot
   never renders and which costs up to 4 Mbps.

Italicised names are design rules, defined in docs/architecture.md.
"""

from __future__ import annotations

import contextlib
from time import monotonic
from typing import TYPE_CHECKING

from pipecat_synthesia._log import get_logger
from pipecat_synthesia.errors import SynthesiaError
from pipecat_synthesia.livekit._rtc import lk_rtc
from pipecat_synthesia.livekit._transport_access import (
    bot_identity,
    check_transport,
    get_room,
    get_token,
    get_url,
    wait_connected,
)
from pipecat_synthesia.livekit._types import (
    LiveKitCredentials,
    MintedAvatarToken,
    ProvidedAvatarToken,
    RoomHandle,
)
from pipecat_synthesia.livekit.audio_filter import AvatarAudioFilter
from pipecat_synthesia.livekit.session import LiveKitAvatarSession
from pipecat_synthesia.livekit.token import (
    DEFAULT_AVATAR_IDENTITY,
    DEFAULT_AVATAR_NAME,
    bot_token_is_agent_kind,
    check_not_expired,
    checked_livekit_url,
    decode_unsigned,
)
from pipecat_synthesia.room import AvatarRoom, JoinMode

if TYPE_CHECKING:
    import loguru
    from pipecat.transports.livekit.transport import LiveKitTransport

    from pipecat_synthesia.livekit._types import AvatarTokenSource
    from pipecat_synthesia.room import AvatarSession, LaunchContext

#: A test asserts this message's text: it names the exact call a developer must add.
BOT_KIND_MESSAGE = (
    'the bot\'s LiveKit token lacks the top-level kind="agent" claim (AccessToken.with_kind("agent")); '
    "the avatar worker would treat the bot as a human and never end the session when the real human "
    "leaves. Pipecat's generate_token_with_agent(), which its dev runner uses for -t livekit, only sets "
    "the video.agent grant, which does not change participant kind."
)

#: Names neither key (*redaction rule*); a test asserts that.
API_KEY_MISMATCH_MESSAGE = (
    "the avatar token will be signed with a different LiveKit API key than the bot's token; if the two "
    "keys belong to different LiveKit projects, the avatar worker cannot join the bot's room, and the "
    "Synthesia API reports that as a 503"
)

_log = get_logger()


class LiveKitTransportRoom(AvatarRoom):
    """Borrows the room the bot's LiveKitTransport is already connected to; the avatar publishes there."""

    join_mode = JoinMode.DIRECT

    def __init__(
        self,
        transport: LiveKitTransport,
        *,
        avatar_token: str | None = None,
        credentials: LiveKitCredentials | None = None,
        livekit_url: str | None = None,
        avatar_identity: str | None = None,
        avatar_name: str = DEFAULT_AVATAR_NAME,
        connect_timeout: float = 30.0,
    ) -> None:
        """Check the transport and the tokens, and settle the avatar's identity and token source.

        A mistake here raises at construction (*misuse vs session errors*), so it surfaces at startup
        rather than seconds into a call. When the avatar token will be minted, a bot token signed with a
        different API key than `credentials` draws one warning: one LiveKit project can have several keys.

        Args:
            transport: The bot's `LiveKitTransport`. Its room is borrowed, never closed.
            avatar_token: A caller-minted avatar token; mutually exclusive with `credentials`.
            credentials: LiveKit API key/secret used to mint the avatar token. Defaults to
                `LiveKitCredentials.from_env()` when neither is given.
            livekit_url: Override the URL sent to Synthesia. Defaults to the transport's own URL,
                which is what the bot is connected to.
            avatar_identity: The avatar's LiveKit identity. Defaults to the `avatar_token`'s `sub`,
                else `DEFAULT_AVATAR_IDENTITY`.
            avatar_name: The avatar's display name.
            connect_timeout: Seconds to wait for Pipecat's own connect.

        Raises:
            TypeError: `transport` is not a `LiveKitTransport`, or its Pipecat internals are not where
                this version expects them.
            ValueError: Both or neither credential route; a bot token that is not a LiveKit access token
                or lacks `kind="agent"`; an `avatar_token` that fails its checks against the bot identity
                and room, or lacks the `canUpdateOwnMetadata` grant; an `avatar_identity` that contradicts
                the token's `sub`; or a URL (`livekit_url`, else the transport's) with no host, or that is
                `ws://` or `http://`.
        """
        check_transport(transport)
        if avatar_token is not None and credentials is not None:
            raise ValueError("pass either avatar_token or credentials, not both")

        # *agent-kind requirement*. The bot token also names the bot identity (`sub`) and its room.
        bot_token = get_token(transport)
        if not bot_token:
            raise ValueError("the LiveKitTransport has no token; pass the bot's LiveKit access token to it")
        try:
            bot_claims = decode_unsigned(bot_token)
        except SynthesiaError:
            raise ValueError("the LiveKitTransport's token is not a LiveKit access token") from None
        if not bot_token_is_agent_kind(bot_token):
            raise ValueError(BOT_KIND_MESSAGE)

        token_identity: str | None = None
        source: AvatarTokenSource
        if avatar_token is not None:
            try:
                provided = ProvidedAvatarToken(
                    avatar_token, bot_identity=bot_claims.identity, room_name=bot_claims.room
                )
            except SynthesiaError as err:
                # a misuse error, so ValueError rather than SynthesiaError
                raise ValueError(str(err)) from err
            token_identity = provided.identity
            source = provided
        else:
            minting = credentials or LiveKitCredentials.from_env()
            if bot_claims.issuer and bot_claims.issuer != minting.api_key:
                _log.warning(API_KEY_MISMATCH_MESSAGE)
            source = MintedAvatarToken(minting)

        if avatar_identity and token_identity and avatar_identity != token_identity:
            raise ValueError(
                f"avatar_identity {avatar_identity!r} does not match the token identity {token_identity!r}"
            )

        # the URL the avatar worker is told to join: the transport's own, unless overridden
        url = checked_livekit_url(
            livekit_url or get_url(transport),
            source="livekit_url" if livekit_url else "the LiveKitTransport's URL",
        )

        self._transport = transport
        # the bot identity and room the avatar token is minted for, before the connect (*connect overlap*)
        self._bot_claims = bot_claims
        self._token_source = source
        self._avatar_identity = avatar_identity or token_identity or DEFAULT_AVATAR_IDENTITY
        self._avatar_name = avatar_name
        self._livekit_url = url
        self._connect_timeout = connect_timeout
        self._filter: AvatarAudioFilter | None = None
        self._filter_setup_seen = False

    @property
    def avatar_identity(self) -> str:
        return self._avatar_identity

    @property
    def filter_installed(self) -> bool:
        """True once the filter from `input_filter()` has had `setup()` called by a pipeline.

        Read by `SynthesiaVideoService.start()`: the developer places this half of the *self-hearing
        guard*, so "never set up" is a real and silent failure, worth one warning.
        """
        return self._filter_setup_seen

    def input_filter(self) -> AvatarAudioFilter:
        """The processor to place right after `transport.input()`; the same instance every call."""
        if self._filter is None:
            self._filter = AvatarAudioFilter(
                avatar_identity=self._avatar_identity, on_setup=self._note_filter_setup
            )
        return self._filter

    def _note_filter_setup(self) -> None:
        self._filter_setup_seen = True

    async def launch(self, ctx: LaunchContext) -> AvatarSession:
        """Launch the avatar into the transport room, and unsubscribe the bot from it."""
        log, claims = ctx.log, self._bot_claims
        # ① A token can expire between startup and launch. Once the bot has connected, its expiry no longer
        # matters: LiveKit checks it only at join.
        if not bot_identity(self._transport):
            check_not_expired(claims, what="the LiveKitTransport's token")

        # ② the session request while Pipecat's own connect runs (*connect overlap*); a bot token that names
        # no room leaves only the connected room to say which, so then the connect comes first
        if claims.room:
            session = await LiveKitAvatarSession.launch_while_connecting(
                self._connect,
                bot_identity=claims.identity,
                room_name=claims.room,
                livekit_url=self._livekit_url,
                token_source=self._token_source,
                avatar_identity=self._avatar_identity,
                avatar_name=self._avatar_name,
                ctx=ctx,
            )
        else:
            log.debug("the bot token names no room, so the session request waits for the connect")
            session = await LiveKitAvatarSession.launch(
                await self._connect(),
                token_source=self._token_source,
                avatar_identity=self._avatar_identity,
                avatar_name=self._avatar_name,
                ctx=ctx,
            )

        # ③ the avatar unsubscribe. A cancellation here (the service's launch timeout) must not leave a
        # live session behind that nobody holds a reference to.
        try:
            await self._unsubscribe_from_avatar(session, session.handle.room, log)
        except BaseException:
            with contextlib.suppress(Exception):
                await session.aclose()
            raise
        return session

    async def _connect(self) -> RoomHandle:
        """Wait for Pipecat's own connect, which runs concurrently with the launch, and borrow its room."""
        await wait_connected(self._transport, timeout=self._connect_timeout)
        connected_at = monotonic()
        room = get_room(self._transport)
        # borrowed: the session never touches this room (*borrowed-room rule*)
        return RoomHandle(
            room=room,
            bot_identity=bot_identity(self._transport),
            livekit_url=self._livekit_url,
            room_name=room.name,
            owns_room=False,
            connected_at=connected_at,
        )

    async def _unsubscribe_from_avatar(
        self, session: AvatarSession, room: lk_rtc.Room, log: loguru.Logger
    ) -> None:
        """The avatar unsubscribe: the bot neither hears nor downloads the avatar.

        The subscription handlers are registered first, so a subscription that lands before the
        immediate calls below is caught by one path or the other; unsubscribing twice is harmless.

        The audio and video unsubscribes then run independently. Both are best effort, but they protect
        different things: by now the avatar's video is normally already subscribed, so a video unsubscribe
        skipped because the audio one raised would never be made up by the handler, and the bot would
        download the whole call's video for nothing.
        """

        async def _on_audio(transport: LiveKitTransport, participant_id: str) -> None:
            if participant_id == self._avatar_identity:
                # awaited: `mute_participant` is `async def`, so a bare call would do nothing
                await transport.mute_participant(participant_id)

        async def _on_video(_transport: LiveKitTransport, participant_id: str) -> None:
            if participant_id == self._avatar_identity:
                self._unsubscribe_video(room, log)

        self._transport.add_event_handler("on_audio_track_subscribed", _on_audio)
        self._transport.add_event_handler("on_video_track_subscribed", _on_video)

        async def _remove_handlers() -> None:
            self._transport.remove_event_handler("on_audio_track_subscribed", _on_audio)
            self._transport.remove_event_handler("on_video_track_subscribed", _on_video)

        session.add_close_hook(_remove_handlers)

        muted = await self._mute_avatar_audio(log)
        unsubscribed = self._unsubscribe_video(room, log)
        if muted and unsubscribed:
            log.info("muted avatar audio and unsubscribed avatar video for the bot")

    async def _mute_avatar_audio(self, log: loguru.Logger) -> bool:
        """Unsubscribe the bot from the avatar's audio; True if the call went through.

        A silent no-op until the avatar's audio track exists; the `on_audio_track_subscribed` handler
        covers a track that arrives later. A failure is logged, not fatal: the avatar audio filter still
        keeps the avatar out of STT, and ending a working session over it would be worse. `CancelledError`
        is not caught: `launch()` closes the session and re-raises.
        """
        try:
            await self._transport.mute_participant(self._avatar_identity)
        except Exception as exc:
            log.warning(f"could not mute the avatar's audio for the bot: {exc!r}")
            return False
        return True

    def _unsubscribe_video(self, room: lk_rtc.Room, log: loguru.Logger) -> bool:
        """`set_subscribed(False)` on every video publication of the avatar (up to 4 Mbps unwatched).

        Returns True when every video publication was dealt with; an avatar that is not in the room
        has nothing to unsubscribe and counts as success.
        """
        avatar = room.remote_participants.get(self._avatar_identity)
        if avatar is None:
            return True
        done = True
        for publication in list(avatar.track_publications.values()):
            if publication.kind == lk_rtc.TrackKind.KIND_VIDEO:
                try:
                    publication.set_subscribed(False)
                except Exception as exc:
                    log.warning(f"could not unsubscribe the avatar's video track: {exc!r}")
                    done = False
        return done
