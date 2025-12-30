from pydantic import BaseModel, Field


class ImageRef(BaseModel):
    id: str = Field(..., pattern=r"^img_\w{6}$")


class AudioRef(BaseModel):
    id: str = Field(..., pattern=r"^aud_\w{6}$")


class VideoRef(BaseModel):
    id: str = Field(..., pattern=r"^vid_\w{6}$")


class DocumentRef(BaseModel):
    id: str = Field(..., pattern=r"^doc_\w{6}$")


class ReconRef(BaseModel):
    id: str = Field(..., pattern=r"^recon_\w{6}$")


class ArrayRef(BaseModel):
    id: str = Field(..., pattern=r"^arr_\w{6}$")


class UrlRef(BaseModel):
    id: str = Field(..., pattern=r"^url_\w{6}$")
