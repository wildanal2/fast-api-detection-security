from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()
  
class FormReq(BaseModel):
    path: str = Field(..., description="Path URL")
    method: str = Field(..., description="Metode HTTP")
    httpVersion: float = Field(..., description="Versi HTTP")
    headers_user_agent: str = Field(..., description="User Agent")
    headers_accept: str = Field(..., description="Header Accept")
    headers_accept_encoding: str = Field(..., description="Header Accept-Encoding")
    remotePort: int = Field(..., description="Remote Port")
    protocol: str = Field(..., description="Protokol")
    duration: int = Field(..., description="Durasi")
    responseStatusCode: int = Field(..., description="Kode Status Respons")
    ip_v4: str = Field(..., description="Alamat IP v4")
    country: str = Field(..., description="Negara")
    isp: str = Field(..., description="Penyedia Layanan Internet")
    org: str = Field(..., description="Organisasi")
    mobile: bool = Field(..., description="Apakah Mobile")
    proxy: bool = Field(..., description="Apakah Proxy")
    hosting: bool = Field(..., description="Apakah Hosting")
    is_abuser: bool = Field(..., description="Apakah Abuser")
    is_attacker: bool = Field(..., description="Apakah Attacker")
    is_bogon: bool = Field(..., description="Apakah Bogon")
    is_cloud_provider: bool = Field(..., description="Apakah Cloud Provider")
    is_proxy: bool = Field(..., description="Apakah Proxy")
    is_relay: bool = Field(..., description="Apakah Relay")
    is_tor: bool = Field(..., description="Apakah Tor")
    is_tor_exit: bool = Field(..., description="Apakah Tor Exit")
    is_vpn: bool = Field(..., description="Apakah VPN")
    is_anonymous: bool = Field(..., description="Apakah Anonim")
    is_threat: bool = Field(..., description="Apakah Threat")
    msElapsedWithIP: int = Field(..., description="Waktu yang Telah Berlalu dengan IP")
    headers_accept_language: str = Field(..., description="Header Accept-Language")
    headers_upgrade_insecure_requests: float = Field(..., description="Header Upgrade-Insecure-Requests")
    headers_connection: str = Field(..., description="Header Connection")
    headers_x_requested_with: str = Field(..., description="Header X-Requested-With")
    headers_content_type: str = Field(..., description="Header Content-Type")
    headers_content_length: float = Field(..., description="Header Content-Length")
    rawBody: str = Field(..., description="Raw Body")

items = []

@app.get("/")
def root():
    return {"status": "up"}

@app.post("/predict")
def prediction_label1(item: FormReq):
    items.append(item)
    return items
 

@app.get("/items/{item_id}", response_model=FormReq)
def get_item(item_id: int) -> str:
    if item_id < len(items):
        return items[item_id]
    else:
        raise HTTPException(status_code=404, detail=f"items {item_id} tidak ditemukan")

