from pydantic import Field
from typing import Annotated


ImageRef = Annotated[str, Field(..., pattern=r"^img_\w{6}$")]
AudioRef = Annotated[str, Field(..., pattern=r"^aud_\w{6}$")]
VideoRef = Annotated[str, Field(..., pattern=r"^vid_\w{6}$")]
DocumentRef = Annotated[str, Field(..., pattern=r"^doc_\w{6}$")]
ReconRef = Annotated[str, Field(..., pattern=r"^recon_\w{6}$")]
ArrayRef = Annotated[str, Field(..., pattern=r"^arr_\w{6}$")]
UrlRef = Annotated[str, Field(..., pattern=r"^url_\w{6}$")]
