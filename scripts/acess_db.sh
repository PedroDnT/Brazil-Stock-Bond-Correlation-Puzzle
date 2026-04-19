## access backend of painelfidc.com.br

curl -v -X OPTIONS \
  'https://ulxfhbyvbjsivbpcmyim.supabase.co/functions/v1/validate-access' \
  -H 'accept: */*' \
  -H 'accept-encoding: gzip, deflate, br, zstd' \
  -H 'accept-language: en-US,en;q=0.9,pt;q=0.8' \
  -H 'access-control-request-headers: apikey,authorization,content-type,x-browser-fingerprint,x-client-info' \
  -H 'access-control-request-method: POST' \
  -H 'origin: https://www.painelfidc.com.br' \
  -H 'referer: https://www.painelfidc.com.br/' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36'

curl -v -X POST \
    'https://ulxfhbyvbjsivbpcmyim.supabase.co/functions/v1/validate-access' \
    -H 'accept: */*' \
    -H 'accept-encoding: gzip, deflate, br, zstd' \
    -H 'accept-language: en-US,en;q=0.9,pt;q=0.8' \
    -H 'content-type: application/json' \
    -H 'origin: https://www.painelfidc.com.br' \
    -H 'referer: https://www.painelfidc.com.br/' \
    -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36'

# print 5 latest lines of access.log
tail -n 5 /var/log/nginx/access.log
